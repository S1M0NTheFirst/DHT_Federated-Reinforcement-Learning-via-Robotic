# worker_client_asr_optimized.py
import os
import gc
import signal
import sys
import psutil # <--- ADDED for Hardware Tracking

# --- 1. SILENCE LOGS (MUST BE FIRST) ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["PYTHONUNBUFFERED"] = "1"

import flwr as fl
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import argparse
import logging
from torch.utils.data import DataLoader
from jiwer import cer
from collections import OrderedDict
from torch.cuda.amp import autocast, GradScaler
import traceback

# --- SETTINGS ---
LIBRISPEECH_ROOT = "/app/data/LibriSpeech/train-clean-100/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_CLASSES = 29
MAX_RETRIES = 10
RETRY_DELAY = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    logger.info(f"[Signal Handler] Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# --- UTILS & DATASET ---
def get_speaker_partitions(num_clients: int):
    """Partition speakers across clients"""
    try:
        ids = sorted([d for d in os.listdir(LIBRISPEECH_ROOT) 
                     if os.path.isdir(os.path.join(LIBRISPEECH_ROOT, d))])
        if not ids:
            raise ValueError(f"No speaker directories found in {LIBRISPEECH_ROOT}")
        partitions = [p.tolist() for p in np.array_split(ids, num_clients)]
        return partitions
    except Exception as e:
        logger.error(f"Error partitioning speakers: {e}")
        raise

def pad_collate_fn(batch):
    """Collate function with error handling"""
    try:
        specs, labels = zip(*batch)
        spec_lengths = torch.tensor([s.shape[2] for s in specs], dtype=torch.long)
        label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
        
        specs_padded = nn.utils.rnn.pad_sequence(
            [s.squeeze(0).permute(1, 0) for s in specs], 
            batch_first=True, 
            padding_value=0.0
        ).permute(0, 2, 1).unsqueeze(1)
        
        labels_padded = nn.utils.rnn.pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=0
        )
        
        return specs_padded, labels_padded, spec_lengths, label_lengths
    except Exception as e:
        logger.error(f"Error in collate function: {e}")
        raise

class TextTransform:
    def __init__(self):
        self.char_map_str = "' abcdefghijklmnopqrstuvwxyz"
        self.char_map = {c: i for i, c in enumerate(self.char_map_str)}
        self.index_map = {i: c for i, c in enumerate(self.char_map_str)}
    
    def text_to_int(self, text): 
        return [self.char_map[c] for c in text.lower() if c in self.char_map]
    
    def int_to_text(self, labels): 
        return "".join([self.index_map.get(i, '') for i in labels])

text_transform = TextTransform()
BLANK_TOKEN_ID = len(text_transform.char_map_str)

class LibriSpeechClientDataset(torch.utils.data.Dataset):
    def __init__(self, speaker_ids, root, transform, text_transform, max_samples=None):
        self.transform = transform
        self.text_transform = text_transform
        self.file_list = []
        self.transcript_map = {}
        
        logger.info(f"Loading dataset for {len(speaker_ids)} speakers...")
        
        for s_id in speaker_ids:
            speaker_path = os.path.join(root, s_id)
            if not os.path.isdir(speaker_path): 
                continue
                
            for c_id in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, c_id)
                if not os.path.isdir(chapter_path): 
                    continue
                    
                transcript_file = os.path.join(chapter_path, f"{s_id}-{c_id}.trans.txt")
                if not os.path.exists(transcript_file): 
                    continue
                
                # Load transcripts
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) != 2: 
                            continue
                        file_id, transcript = parts
                        cleaned = "".join(filter(lambda c: c in text_transform.char_map_str, 
                                                transcript.lower()))
                        if cleaned:  # Only add if transcript is not empty
                            self.transcript_map[file_id] = cleaned
                
                # Load audio files
                for f_name in os.listdir(chapter_path):
                    if f_name.endswith(".flac"):
                        file_id = f_name.replace(".flac", "")
                        if file_id in self.transcript_map:  # Only add if we have transcript
                            self.file_list.append(os.path.join(chapter_path, f_name))
        
        if max_samples and len(self.file_list) > max_samples:
            self.file_list = self.file_list[:max_samples]
        
        logger.info(f"Dataset loaded: {len(self.file_list)} samples")
    
    def __len__(self): 
        return len(self.file_list)
    
    def __getitem__(self, idx):
        try:
            audio_path = self.file_list[idx]
            waveform, _ = torchaudio.load(audio_path)
            spectrogram = self.transform(waveform)
            
            file_id = os.path.basename(audio_path).replace(".flac", "")
            transcript = self.transcript_map.get(file_id, "")
            
            if not transcript:
                logger.warning(f"Empty transcript for {file_id}")
                transcript = " "  # Fallback to space
            
            label = torch.tensor(self.text_transform.text_to_int(transcript), dtype=torch.long)
            
            return spectrogram, label
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return a dummy sample instead of crashing
            dummy_spec = torch.zeros(1, 80, 100)
            dummy_label = torch.tensor([0], dtype=torch.long)
            return dummy_spec, dummy_label

# --- MODEL ---
class SimpleASR(nn.Module):
    def __init__(self, in_feat, rnn_h, n_class, down_factor=4, dropout=0.1):
        super(SimpleASR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True), 
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 32, 3, 2, 1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True), 
            nn.Dropout2d(dropout)
        )
        self.rnn_input_size = 32 * (in_feat // down_factor)
        self.rnn = nn.GRU(
            self.rnn_input_size, rnn_h, 2, 
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.fc = nn.Linear(rnn_h * 2, n_class)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1, x.size(3)).transpose(1, 2)
        x, _ = self.rnn(x)
        return self.fc(x)

def greedy_decoder(output, labels, label_lengths, blank_label):
    """Decode CTC output to text"""
    arg_maxes = torch.argmax(output, dim=2)
    decoded_preds, decoded_targets = [], []
    
    for i, args in enumerate(arg_maxes):
        decoded_pred = []
        for j, index in enumerate(args):
            if index != blank_label:
                if j == 0 or index != args[j - 1]:
                    decoded_pred.append(index.item())
        
        decoded_preds.append(text_transform.int_to_text(decoded_pred))
        decoded_targets.append(
            text_transform.int_to_text(labels[i][:label_lengths[i]].tolist())
        )
    
    return decoded_preds, decoded_targets

# --- TRAINING ---
def test(model, data_loader):
    """Evaluation with memory cleanup"""
    criterion = nn.CTCLoss(blank=BLANK_TOKEN_ID, zero_infinity=True)
    model.eval()
    model.to(DEVICE)
    
    total_loss = 0.0
    total_cer = 0.0
    total_samples = 0
    start_time = time.time() # <--- Added Timer
    
    try:
        with torch.no_grad():
            for batch_idx, (data, target, spec_lengths, target_lengths) in enumerate(data_loader):
                if shutdown_requested:
                    logger.info("Shutdown requested during evaluation")
                    break
                
                data = data.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)
                
                with autocast():
                    output = model(data)
                    output_lengths = torch.div(spec_lengths, 4, rounding_mode='floor')
                    log_probs = F.log_softmax(output, dim=2).permute(1, 0, 2)
                    loss = criterion(log_probs, target, output_lengths, target_lengths)
                
                total_loss += loss.item()
                
                decoded_preds, decoded_targets = greedy_decoder(
                    output.cpu(), target.cpu(), target_lengths, BLANK_TOKEN_ID
                )
                
                if decoded_targets:  # Avoid division by zero
                    total_cer += cer(decoded_targets, decoded_preds)
                
                total_samples += 1
                
                # Cleanup
                del data, target, output, log_probs, loss
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_accuracy = 1 - (total_cer / total_samples) if total_samples > 0 else 0
        
        return avg_loss, avg_accuracy, (time.time() - start_time) # <--- Return Duration
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return 0.0, 0.0, 0.0
    finally:
        torch.cuda.empty_cache()
        gc.collect()

def train(model, train_loader, epochs, proximal_mu=0.0):
    """Training with memory management and error handling"""
    criterion = nn.CTCLoss(blank=BLANK_TOKEN_ID, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, 
        epochs=epochs, 
        steps_per_epoch=len(train_loader)
    )
    scaler = GradScaler()
    model.to(DEVICE)
    
    # --- FIX FOR FEDPROX: Clone params properly ---
    global_params = [p.detach().clone() for p in model.parameters()] if proximal_mu > 0 else None
    
    last_loss = 0.0
    start_time = time.time() # <--- Added Timer
    
    try:
        for epoch in range(epochs):
            if shutdown_requested:
                logger.info(f"Shutdown requested during epoch {epoch}")
                break
            
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, (data, target, spec_lengths, target_lengths) in enumerate(train_loader):
                if shutdown_requested:
                    break
                
                data = data.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with autocast():
                    output = model(data)
                    output_lengths = torch.div(spec_lengths, 4, rounding_mode='floor')
                    log_probs = F.log_softmax(output, dim=2).permute(1, 0, 2)
                    loss = criterion(log_probs, target, output_lengths, target_lengths)
                    
                    # --- FIX FOR FEDPROX: Correct Calculation ---
                    if proximal_mu > 0.0 and global_params is not None:
                        prox_loss = 0.0
                        for param, g_param in zip(model.parameters(), global_params):
                            prox_loss += (param - g_param).pow(2).sum() # Use pow(2).sum() instead of norm
                        loss += (proximal_mu / 2) * prox_loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                last_loss = loss.item()
                
                # Cleanup every few batches
                if batch_idx % 5 == 0:
                    del data, target, output, log_probs, loss
                    torch.cuda.empty_cache()
            
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            logger.info(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}")
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())
    
    finally:
        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    # Return Duration
    return last_loss, (time.time() - start_time)

# --- CLIENT ---
class ASRClient(fl.client.NumPyClient):
    def __init__(self, model, client_id, train_loader, val_loader):
        self.model = model
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Initial Net snapshot
        self.net_start = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv

    def get_parameters(self, config):
        """Extract model parameters"""
        try:
            return [p.cpu().numpy() for _, p in self.model.state_dict().items()]
        except Exception as e:
            logger.error(f"Error getting parameters: {e}")
            raise

    def set_parameters(self, params):
        """Set model parameters"""
        try:
            params_dict = zip(self.model.state_dict().keys(), params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            logger.error(f"Error setting parameters: {e}")
            raise

    def fit(self, params, config):
        """Train the model"""
        try:
            self.set_parameters(params)
            epochs = int(config.get("local_epochs", 1))
            mu = float(config.get("proximal_mu", 0.0))
            
            logger.info(f"[Client {self.client_id}] Starting training for {epochs} epochs...")
            
            # --- CAPTURE TRACKING DATA ---
            loss, duration = train(self.model, self.train_loader, epochs, mu)
            
            cpu_usage = psutil.cpu_percent()
            gpu_usage = 0.0
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated(DEVICE) / (1024 * 1024) # MB

            # Network Delta
            current_net = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
            network_mb = (current_net - self.net_start) / (1024 * 1024)
            self.net_start = current_net # Reset for next round

            logger.info(f"[Client {self.client_id}] Training complete. Loss: {loss:.4f}")
            
            return self.get_parameters({}), len(self.train_loader.dataset), {
                "train_loss": float(loss),
                "train_time": float(duration),
                "cpu_usage": float(cpu_usage),
                "gpu_usage": float(gpu_usage),
                "network_mb": float(network_mb)
            }
        
        except Exception as e:
            logger.error(f"[Client {self.client_id}] Error in fit: {e}")
            logger.error(traceback.format_exc())
            raise

    def evaluate(self, params, config):
        """Evaluate the model"""
        try:
            self.set_parameters(params)
            loss, accuracy, duration = test(self.model, self.val_loader) # Receive duration
            
            cpu_usage = psutil.cpu_percent()
            gpu_usage = 0.0
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated(DEVICE) / (1024 * 1024)

            logger.info(f"[Client {self.client_id}] Eval: Acc={accuracy:.4f}, Loss={loss:.4f}")
            
            return float(loss), len(self.val_loader.dataset), {
                "accuracy": float(accuracy), 
                "loss": float(loss),
                "eval_time": float(duration),
                "cpu_usage": float(cpu_usage),
                "gpu_usage": float(gpu_usage)
            }
        
        except Exception as e:
            logger.error(f"[Client {self.client_id}] Error in evaluate: {e}")
            logger.error(traceback.format_exc())
            return 0.0, 0, {"accuracy": 0.0, "loss": 0.0}

def cleanup_resources():
    """Clean up GPU and system resources"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--num-clients", type=int, default=8)
    args = parser.parse_args()
    
    SERVER = os.getenv("MASTER_ADDRESS", "127.0.0.1:8080")
    logger.info(f"[Client {args.client_id}] Initializing... Server: {SERVER}")
    
    try:
        # Load Data
        parts = get_speaker_partitions(args.num_clients)[args.client_id]
        xform = torchaudio.transforms.MelSpectrogram(n_mels=80)
        
        full_dataset = LibriSpeechClientDataset(
            parts, LIBRISPEECH_ROOT, xform, text_transform, max_samples=5000
        )
        
        train_size = int(0.8 * len(full_dataset))
        train_sub, val_sub = torch.utils.data.random_split(
            full_dataset, 
            [train_size, len(full_dataset) - train_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # Reduced batch size and workers to prevent OOM
        train_loader = DataLoader(
            train_sub, batch_size=8, shuffle=True, 
            collate_fn=pad_collate_fn, 
            num_workers=2, pin_memory=True, 
            prefetch_factor=2, persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_sub, batch_size=8, 
            collate_fn=pad_collate_fn, 
            num_workers=2, pin_memory=True, 
            prefetch_factor=2, persistent_workers=True
        )
        
        logger.info(f"[Client {args.client_id}] Data loaded successfully")
        
        methods = ["FedAvg", "FedProx", "FedAdam"]
        
        for i, method in enumerate(methods):
            if shutdown_requested:
                logger.info("Shutdown requested, exiting...")
                break
            
            logger.info(f"\n{'='*40}")
            logger.info(f"[Client {args.client_id}] STARTING {method} ({i+1}/3)")
            logger.info(f"{'='*40}")
            
            method_completed = False
            retry_count = 0
            
            while not method_completed and retry_count < MAX_RETRIES:
                if shutdown_requested:
                    break
                
                try:
                    # Re-initialize model for each method
                    model = SimpleASR(in_feat=80, rnn_h=256, n_class=N_CLASSES, dropout=0.1)
                    client = ASRClient(model, args.client_id, train_loader, val_loader)
                    
                    logger.info(f"[Client {args.client_id}] Connecting to server (attempt {retry_count + 1}/{MAX_RETRIES})...")
                    
                    # Start client with proper conversion
                    fl.client.start_client(
                        server_address=SERVER,
                        client=client.to_client()
                    )
                    
                    logger.info(f"[Client {args.client_id}] Finished {method}")
                    method_completed = True
                    
                    # Cleanup between methods
                    del model, client
                    cleanup_resources()
                
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received")
                    shutdown_requested = True
                    break
                
                except Exception as e:
                    retry_count += 1
                    logger.warning(
                        f"[Client {args.client_id}] Connection failed ({e}). "
                        f"Retry {retry_count}/{MAX_RETRIES} in {RETRY_DELAY}s..."
                    )
                    cleanup_resources()
                    time.sleep(RETRY_DELAY)
            
            if not method_completed and not shutdown_requested:
                logger.error(f"[Client {args.client_id}] Failed to complete {method} after {MAX_RETRIES} retries")
            
            # Wait before next method
            if i < len(methods) - 1 and not shutdown_requested:
                logger.info(f"[Client {args.client_id}] Waiting 20s for next method...")
                time.sleep(20)
        
        logger.info(f"[Client {args.client_id}] All methods completed")
    
    except Exception as e:
        logger.error(f"[Client {args.client_id}] Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    finally:
        cleanup_resources()
        logger.info(f"[Client {args.client_id}] Shutting down gracefully")
        sys.exit(0)