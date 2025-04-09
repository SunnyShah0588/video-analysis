# Video Analysis and Search System - Installation Guide

This guide will help you set up and run the Video Analysis and Search System on your M1 Mac.

## Prerequisites

- macOS running on Apple Silicon (M1/M2/M3)
- Python 3.8+ installed (Homebrew version recommended)
- pip (Python package manager)
- At least 2GB of free disk space for models and dependencies

## Installation

1. Create a new Python virtual environment:

```bash
# Create a directory for the project
mkdir video-analysis
cd video-analysis

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

2. Install the required dependencies:

```bash
# Install PyTorch with MPS support (optimized for M1 Macs)
pip install torch torchvision

# Install other dependencies
pip install transformers faiss-cpu pillow opencv-python tqdm numpy
```

3. Save the code from the "Video Analysis and Search Package" artifact to a file named `video_embeddings.py` in your project directory.

## Usage

### Extracting Frames and Building an Index

To extract frames from a video and build a searchable index:

```bash
python video_embeddings.py extract /path/to/your/video.mp4 --interval 0.5 --output video_index
```

Parameters:
- `extract`: Command to extract frames and build an index
- `/path/to/your/video.mp4`: Path to your video file
- `--interval 0.5`: Extract a frame every 0.5 seconds (adjust as needed)
- `--output video_index`: Path to save the embedding index

This process will:
1. Extract frames from the video at the specified interval
2. Generate embeddings for each frame using CLIP
3. Build a FAISS index for efficient similarity search
4. Save the index and metadata to disk

### Searching with Text or Images

Once you've built an index, you can search for similar frames using text or image queries:

#### Text Search

```bash
python video_embeddings.py search --model video_index --text "a person riding a bicycle" --results 5 --output results_folder
```

Parameters:
- `search`: Command to search the index
- `--model video_index`: Path to the previously saved index
- `--text "a person riding a bicycle"`: Text query to search for
- `--results 5`: Number of results to return
- `--output results_folder`: Directory to save the result frames

#### Image Search

```bash
python video_embeddings.py search --model video_index --image query.jpg --results 5 --output results_folder
```

Parameters:
- `--image query.jpg`: Path to an image to use as a query

## Example Workflow

Here's a complete example workflow:

```bash
# Activate your virtual environment
source venv/bin/activate

# Extract frames and build index from your video
python video_embeddings.py extract ~/Videos/vacation.mp4 --interval 1.0 --output vacation_index

# Search for frames with a beach
python video_embeddings.py search --model vacation_index --text "beautiful beach with waves" --results 10 --output beach_scenes

# Search for frames similar to a reference image
python video_embeddings.py search --model vacation_index --image reference_sunset.jpg --results 5 --output sunset_scenes
```

## Performance Tips

1. **Frame Interval**: Adjust the `--interval` parameter based on your video's length and content. Shorter intervals provide more comprehensive coverage but increase processing time and storage requirements.

2. **MPS Acceleration**: The code automatically uses Apple's Metal Performance Shaders (MPS) for hardware acceleration on M1 Macs. Ensure you have the latest macOS for optimal performance.

3. **Memory Usage**: For very long videos, you may need to process them in chunks to avoid memory issues. Consider splitting long videos into smaller segments.

4. **Storage Optimization**: The extracted frames are stored in a temporary directory. You can modify the code to save disk space by using lower JPEG quality or resizing frames.

## Troubleshooting

If you encounter issues:

1. **CLIP Model Download**: The first run will download the CLIP model, which may take some time. Ensure you have a stable internet connection.

2. **Missing Dependencies**: If you encounter errors about missing modules, install them with `pip install [module_name]`.

3. **Memory Errors**: If you see out-of-memory errors, reduce the batch size in the code or process a shorter video segment.

4. **FAISS Issues**: If FAISS installation fails, try installing it with conda: `conda install -c conda-forge faiss-cpu`.

5. **MPS Acceleration**: If you experience issues with MPS, you can force CPU usage by modifying the code to set `device="cpu"`.