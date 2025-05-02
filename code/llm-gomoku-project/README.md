# Gomoku GPT-2 Fine-Tuning Pipeline

This project implements a pipeline for fine-tuning GPT-2 models to play Gomoku (Five in a Row), based on the implementation plan in the parent project.

## Project Structure

```
llm-gomoku-project/
├── data/               # Data directory
│   ├── raw/            # Raw game data
│   └── processed/      # Processed datasets
├── src/                # Source code
│   ├── schema.py       # Data schema definitions
│   ├── data_processor.py  # Data processing utilities
│   ├── trainer.py      # Model training code
│   ├── evaluate.py     # Evaluation utilities
│   └── inference.py    # Inference and demo
├── models/             # Saved model checkpoints
├── configs/            # Configuration files
└── requirements.txt    # Project dependencies
```

## Setup

1. Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Processing

The data processing pipeline converts raw Gomoku game data into a format suitable for training GPT-2:

```bash
# Process data
mkdir -p data/raw data/processed
# Copy your raw Gomoku game data into the data/raw directory
python src/data_processor.py
```

## Training

Train a GPT-2 model using LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning:

```bash
python src/trainer.py \
  --model gpt2-medium \
  --data_dir data/processed/gomoku_dataset \
  --output_dir models \
  --epochs 3 \
  --batch_size 8 \
  --lr 3e-5 \
  --r 8 \
  --lora_alpha 32 \
  --frozen_layers 12
```

Training configurations:

- **P1 (Default)**: 12 frozen layers, LoRA r=8/α=32, LR=3e-5, 3 epochs
- **P2**: 12 frozen layers, LoRA r=16/α=32, LR=1e-5, 5 epochs
- **P3**: 6 frozen layers, LoRA r=8/α=16, LR=3e-5, 3 epochs

## Evaluation

Evaluate the model's performance on the test set:

```bash
python src/evaluate.py \
  --model_path models/gomoku-gpt2_final \
  --data_dir data/processed/gomoku_dataset \
  --output_dir evaluation_results \
  --split test
```

## Inference

Use the model to predict moves for Gomoku game states:

```bash
# Run with a sample board
python src/inference.py --model_path models/gomoku-gpt2_final

# Run in interactive mode
python src/inference.py --model_path models/gomoku-gpt2_final --interactive
```

## Data Format

Gomoku game data is represented as:
- Each board state is a grid where `.` represents empty, `X` represents black, and `O` represents white
- Moves are represented in algebraic notation (e.g., "H8")
- The model is trained to predict the next move given a board state

## Parameter-Efficient Fine-Tuning

The project uses several techniques to make training efficient:
- LoRA (Low-Rank Adaptation) to reduce trainable parameters
- Layer freezing (freezing bottom transformer blocks)
- Gradient checkpointing to reduce memory usage
- Mixed precision training (FP16)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 


#ai
I've created a comprehensive pipeline for fine-tuning GPT-2 for Gomoku game decision-making, following the implementation plan. Here's what the pipeline includes:

1. Project Structure:

- Organized directories for data, source code, models, and configs

- Requirements file with all necessary dependencies

2. Data Pipeline:

- Pydantic schema for game data

- Data processor to convert raw game data to training format

- Data generator for synthetic Gomoku games

3. Training Infrastructure:

- Parameter-efficient fine-tuning with LoRA

- Layer freezing and gradient checkpointing

- Integration with Weights & Biases for experiment tracking

- DeepSpeed configuration for distributed training

4. Evaluation Framework:

- Metrics calculation (accuracy, exact match)

- Detailed evaluation reporting

5. Inference:

- Model serving for predictions

- Interactive demo for testing the trained model

6. Complete Pipeline Script:

- Single command to run the entire workflow

To use this pipeline:



```bash
# Install dependencies
pip install -r llm-gomoku-project/requirements.txt

# Run the complete pipeline
cd llm-gomoku-project
python run_pipeline.py --all

# Or run individual steps
python run_pipeline.py --generate_data --num_games 1000
python run_pipeline.py --process_data
python run_pipeline.py --train --model gpt2-medium --epochs 3
python run_pipeline.py --evaluate
python run_pipeline.py --demo
```

The pipeline implements all the requirements from the implementation plan, including parameter-efficient methods (LoRA, layer freezing), configurable training setups, and evaluation metrics.