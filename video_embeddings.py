import os
import cv2
import torch
import faiss
import numpy as np
import argparse
import pickle
import tempfile
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from transformers import CLIPProcessor, CLIPModel

class VideoEmbeddingSystem:
    """System for extracting frames, generating embeddings, and searching videos."""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """
        Initialize the video embedding system with CLIP model.
        
        Args:
            model_name: Name of the CLIP model to use
            device: Device to use (None for auto-detection)
        """
        # Auto-detect device (uses MPS on M1 Macs if available)
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load CLIP model for embeddings
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Initialize FAISS index (will be created when embeddings are generated)
        self.index = None
        self.frame_data = []
        
    def extract_frames(self, video_path: str, frame_interval: float = 1.0) -> List[Dict[str, Any]]:
        """
        Extract frames from a video at specified intervals.
        
        Args:
            video_path: Path to the video file
            frame_interval: Interval in seconds between frames
            
        Returns:
            List of dictionaries with frame data
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"FPS: {fps}")
        print(f"Total frames: {frame_count}")
        
        # Calculate frame indices to extract
        frame_indices = []
        current_time = 0
        while current_time < duration:
            frame_indices.append(int(current_time * fps))
            current_time += frame_interval
            
        print(f"Extracting {len(frame_indices)} frames at {frame_interval}s intervals")
        
        # Extract frames
        frames_data = []
        
        for i, frame_idx in enumerate(tqdm(frame_indices, desc="Extracting frames")):
            # Set the position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Read the frame
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Could not read frame at index {frame_idx}")
                continue
                
            # Calculate timestamp
            timestamp = frame_idx / fps
            
            # Create temporary file for the frame
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
                frame_path = temp.name
                
            # Save frame to file
            cv2.imwrite(frame_path, frame)
            
            # Add to frames data
            frames_data.append({
                "frame_id": i,
                "timestamp": timestamp,
                "path": frame_path,
                "video_path": video_path
            })
            
        cap.release()
        print(f"Successfully extracted {len(frames_data)} frames")
        
        return frames_data
        
    def generate_embeddings(self, frames_data: List[Dict[str, Any]], batch_size: int = 8) -> np.ndarray:
        """
        Generate embeddings for frames using CLIP.
        
        Args:
            frames_data: List of frame data dictionaries
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        all_embeddings = []
        
        # Process frames in batches
        for i in range(0, len(frames_data), batch_size):
            batch = frames_data[i:i+batch_size]
            
            # Load images
            images = []
            for frame in batch:
                image = Image.open(frame["path"])
                images.append(image)
                
            # Process images with CLIP
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            
            # Generate image embeddings
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                
            # Normalize embeddings
            embeddings = outputs.cpu().numpy()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            all_embeddings.append(embeddings)
            
        # Concatenate all batches
        embeddings_array = np.vstack(all_embeddings)
        
        print(f"Generated embeddings with shape: {embeddings_array.shape}")
        
        # Store frame data
        self.frame_data = frames_data
        
        # Create FAISS index
        self._build_index(embeddings_array)
        
        return embeddings_array
        
    def _build_index(self, embeddings: np.ndarray):
        """Build a FAISS index from embeddings."""
        dimension = embeddings.shape[1]
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product index for cosine similarity
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        print(f"Built FAISS index with {self.index.ntotal} vectors")
        
    def search(self, query_image=None, query_text=None, k=5) -> List[Dict[str, Any]]:
        """
        Search for similar frames using an image or text query.
        
        Args:
            query_image: Path to query image (optional)
            query_text: Text query (optional)
            k: Number of results to return
            
        Returns:
            List of matching frame data
        """
        if query_image is None and query_text is None:
            raise ValueError("Either query_image or query_text must be provided")
            
        if self.index is None:
            raise ValueError("No index available. Generate embeddings first.")
            
        # Generate query embedding
        if query_image is not None:
            # Load and process image
            image = Image.open(query_image)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate image embedding
            with torch.no_grad():
                query_embedding = self.model.get_image_features(**inputs)
        else:
            # Process text
            inputs = self.processor(text=query_text, return_tensors="pt").to(self.device)
            
            # Generate text embedding
            with torch.no_grad():
                query_embedding = self.model.get_text_features(**inputs)
                
        # Convert to numpy and normalize
        query_embedding = query_embedding.cpu().numpy()
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Process results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.frame_data):  # Ensure valid index
                frame = self.frame_data[idx].copy()  # Copy to avoid modifying original
                frame["similarity"] = float(distances[0][i])
                results.append(frame)
                
        return results
        
    def save(self, path: str):
        """Save the system state to a file."""
        if self.index is None:
            raise ValueError("No index available to save")
            
        state = {
            "frame_data": self.frame_data
        }
        
        # Save state
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(state, f)
            
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        print(f"Saved system state to {path}.pkl and {path}.index")
        
    def load(self, path: str):
        """Load the system state from a file."""
        # Load state
        with open(f"{path}.pkl", "rb") as f:
            state = pickle.load(f)
            
        self.frame_data = state["frame_data"]
        
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.index")
        
        print(f"Loaded system with {len(self.frame_data)} frames and {self.index.ntotal} vectors")
        

def main():
    parser = argparse.ArgumentParser(description="Video Embedding System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Extract frames command
    extract_parser = subparsers.add_parser("extract", help="Extract frames from a video")
    extract_parser.add_argument("video_path", help="Path to the video file")
    extract_parser.add_argument("--interval", type=float, default=1.0, help="Interval between frames in seconds")
    extract_parser.add_argument("--output", required=True, help="Output path for the embedding system")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar frames")
    search_parser.add_argument("--model", help="Path to the saved embedding system")
    search_parser.add_argument("--image", help="Path to query image")
    search_parser.add_argument("--text", help="Text query")
    search_parser.add_argument("--results", type=int, default=5, help="Number of results to return")
    search_parser.add_argument("--output", help="Directory to save result frames")
    
    args = parser.parse_args()
    
    if args.command == "extract":
        # Create system
        system = VideoEmbeddingSystem()
        
        # Extract frames
        frames_data = system.extract_frames(args.video_path, args.interval)
        
        # Generate embeddings
        system.generate_embeddings(frames_data)
        
        # Save system
        system.save(args.output)
        
    elif args.command == "search":
        if not args.model:
            parser.error("--model is required for search")
            
        if not args.image and not args.text:
            parser.error("Either --image or --text must be provided")
            
        # Load system
        system = VideoEmbeddingSystem()
        system.load(args.model)
        
        # Search
        results = system.search(args.image, args.text, args.results)
        
        # Display results
        print(f"\nSearch results:")
        for i, result in enumerate(results):
            print(f"{i+1}. Timestamp: {result['timestamp']:.2f}s, Similarity: {result['similarity']:.4f}")
            
            # Save or display result frames
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                output_path = os.path.join(args.output, f"result_{i+1}.jpg")
                # Copy the frame to output directory
                import shutil
                shutil.copy(result["path"], output_path)
                print(f"   Saved to: {output_path}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
