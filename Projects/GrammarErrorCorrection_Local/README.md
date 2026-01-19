# Grammar Error Correction - Local Setup

This is a modified version of the Grammar Error Correction notebook, configured to run locally with GPU support.

## Directory Structure

```
GrammarErrorCorrection_Local/
├── GEC/
│   ├── data/                           # Dataset files
│   │   ├── lang8.train.auto.bea19.m2   # Extracted dataset
│   │   ├── corrected.txt               # Generated during preprocessing
│   │   ├── error.txt                   # Generated during preprocessing
│   │   ├── raw_data.csv                # Intermediate data
│   │   ├── preprocessed_15.csv         # Preprocessed data
│   │   └── final_preprocessed_15.csv   # Final training data
│   ├── embeddings/                     # FastText embeddings
│   │   ├── wiki-news-300d-1M.vec.zip   # Downloaded automatically
│   │   ├── wiki-news-300d-1M.vec       # Extracted embeddings
│   │   ├── in_embedding.npy            # Input vocab embeddings
│   │   └── out_embedding.npy           # Output vocab embeddings
│   ├── models/
│   │   ├── ENC_DEC_EMB/                # Model weights (with FastText)
│   │   │   ├── weights_*.weights.h5    # Checkpoint files
│   │   │   └── history.csv             # Training history
│   │   ├── ENC_DEC/                    # Model weights (without FastText)
│   │   └── logs/                       # TensorBoard logs
│   └── EDA and Encoder-Decoder models in various configurations.ipynb
├── requirements.txt
└── README.md
```

## Prerequisites

### 1. Install Dependencies

```bash
cd GrammarErrorCorrection_Local
pip install -r requirements.txt
```

### 2. GPU Setup (Recommended)

For TensorFlow GPU support, you need:
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** (version compatible with your TensorFlow version)
- **cuDNN** library

Check TensorFlow-CUDA compatibility: https://www.tensorflow.org/install/source#gpu

Verify GPU is detected:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### 3. Download Dataset

Place `lang8.bea19.tar.gz` in your **Downloads folder** (`~/Downloads/`).

The dataset is from the BEA 2019 Shared Task on Grammatical Error Correction.
- Source: https://www.cl.cam.ac.uk/research/nl/bea2019st/

### 4. FastText Embeddings

The notebook will automatically download FastText embeddings (~650MB) on first run.
- Source: https://fasttext.cc/docs/en/english-vectors.html

## Running the Notebook

1. Open the notebook in Jupyter:
   ```bash
   cd GrammarErrorCorrection_Local/GEC
   jupyter notebook "EDA and Encoder-Decoder models in various configurations.ipynb"
   ```

2. Run cells sequentially from the top.

3. The first run will:
   - Extract the dataset from Downloads
   - Download FastText embeddings
   - Process and save intermediate files

## Key Changes from Colab Version

| Original (Colab) | Local Version |
|------------------|---------------|
| Google Drive mount | Local file paths |
| `/content/` paths | `./data/`, `./embeddings/`, `./models/` |
| `!pip install` | Use `requirements.txt` |
| `!wget`, `!unzip` | Python `urllib`, `zipfile` |
| `!mkdir`, `!cp` | Python `os.makedirs`, `shutil` |

## Troubleshooting

### GPU Not Detected
- Ensure CUDA and cuDNN are properly installed
- Check TensorFlow version compatibility with your CUDA version
- Try: `pip install tensorflow[and-cuda]` for bundled CUDA support

### Out of Memory (OOM)
- Reduce `BATCH_SIZE` in the notebook
- Reduce `NUMBER_OF_DATAPOINTS` for testing
- Enable memory growth: `tf.config.experimental.set_memory_growth(gpu, True)`

### Dataset Not Found
- Ensure `lang8.bea19.tar.gz` is in `~/Downloads/`
- Or update `DOWNLOADS_DIR` in Cell 2 to point to your file location

## Training Notes

- Training with 10,000 datapoints takes ~30-60 minutes on GPU
- Full dataset (288k samples) requires significant time and memory
- Model weights are saved after each epoch to `models/ENC_DEC_EMB/`
