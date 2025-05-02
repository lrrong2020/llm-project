import os
import argparse
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

class GomokuEvaluator:
    def __init__(
        self,
        model_path: str,
        data_dir: str,
        output_dir: str = "evaluation_results",
        split: str = "test",
    ):
        """
        Initialize the Gomoku model evaluator.
        
        Args:
            model_path: Path to the fine-tuned model
            data_dir: Directory containing processed dataset
            output_dir: Directory to save evaluation results
            split: Dataset split to evaluate on
        """
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.split = split
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        try:
            if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
                # This is a PEFT/LoRA model
                base_model_path = self._get_base_model_path()
                print(f"Loading base model from {base_model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.float32,
                    device_map="auto",
                )
                print(f"Loading adapter from {self.model_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.model_path,
                    torch_dtype=torch.float32,
                    device_map="auto",
                )
            else:
                # Standard model
                print(f"Loading standard model from {self.model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    device_map="auto",
                )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying with different configuration...")
            # Fallback to simpler loading parameters
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path if isinstance(self.model_path, str) else str(self.model_path),
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            print("Model loaded with fallback method")
        
        # No need for external metrics library
        print("Evaluation initialized successfully")
        
    def _get_base_model_path(self) -> str:
        """Get base model path from adapter config."""
        config_path = os.path.join(self.model_path, "adapter_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get("base_model_name_or_path", "gpt2-medium")
    
    def load_dataset(self):
        """Load the evaluation dataset."""
        self.dataset = load_from_disk(self.data_dir)
        assert self.split in self.dataset, f"Split '{self.split}' not found in dataset"
        self.eval_dataset = self.dataset[self.split]
        print(f"Loaded {len(self.eval_dataset)} examples for evaluation")
        
    def preprocess_examples(self, num_examples: Optional[int] = None):
        """Preprocess examples for evaluation."""
        examples = []
        if num_examples is not None:
            eval_dataset = self.eval_dataset.select(range(min(num_examples, len(self.eval_dataset))))
        else:
            eval_dataset = self.eval_dataset
        
        for ex in tqdm(eval_dataset):
            prompt = f"<board>{ex['state']}</board><move>"
            reference = ex['decision']
            example = {
                "game_id": ex["game_id"],
                "turn_idx": ex["turn_idx"],
                "prompt": prompt,
                "reference": reference,
            }
            examples.append(example)
        return examples
    
    def generate_predictions_old(
        self,
        examples: List[Dict],
        max_new_tokens: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> List[Dict]:
        """Generate predictions for evaluation examples."""
        results = []
        
        self.model.eval()
        with torch.no_grad():
            for ex in tqdm(examples):
                try:
                    prompt = ex["prompt"]
                    reference = ex["reference"]
                    
                    # Tokenize the prompt
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    
                    # Generate prediction
                    output = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    # Decode prediction, remove the prompt part
                    prediction_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
                    
                    # Safely extract the move part
                    try:
                        if "<move>" in prediction_text and "</move>" in prediction_text:
                            prediction = prediction_text.split("<move>")[1].split("</move>")[0].strip()
                        else:
                            # Fallback if tags aren't found
                            prediction = prediction_text.replace(prompt, "").strip()
                    except Exception as e:
                        print(f"Error extracting prediction: {e}")
                        prediction = "ERROR"
                    
                    # Store results
                    results.append({
                        "game_id": ex["game_id"],
                        "turn_idx": ex["turn_idx"],
                        "prompt": prompt,
                        "reference": reference,
                        "prediction": prediction,
                        "matched": prediction == reference,
                    })
                except Exception as e:
                    print(f"Error processing example: {e}")
                    # Add a placeholder for the failed example
                    results.append({
                        "game_id": ex.get("game_id", "unknown"),
                        "turn_idx": ex.get("turn_idx", -1),
                        "prompt": ex.get("prompt", ""),
                        "reference": ex.get("reference", ""),
                        "prediction": "ERROR",
                        "matched": False,
                    })
        
        return results
    
    def generate_predictions(
        self,
        examples: List[Dict],
        max_new_tokens: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,  # Enable sampling by default
    ) -> List[Dict]:
        """Generate predictions for evaluation examples."""
        results = []
        
        self.model.eval()
        with torch.no_grad():
            for ex in tqdm(examples):
                try:
                    prompt = ex["prompt"]
                    reference = ex["reference"]
                    
                    # Tokenize the prompt with explicit attention mask
                    inputs = self.tokenizer(
                        prompt, 
                        return_tensors="pt",
                        padding=True,
                        return_attention_mask=True
                    ).to(self.model.device)
                    
                    # Generate prediction with sampling enabled
                    output = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,  # Enable sampling
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    # Decode prediction, remove the prompt part
                    prediction_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
                    
                    # Safely extract the move part
                    try:
                        if "<move>" in prediction_text and "</move>" in prediction_text:
                            prediction = prediction_text.split("<move>")[1].split("</move>")[0].strip()
                        else:
                            # Fallback if tags aren't found
                            prompt_len = len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))
                            prediction = prediction_text[prompt_len:].strip()
                    except Exception as e:
                        print(f"Error extracting prediction: {e}")
                        prediction = "ERROR"
                    
                    # Store results
                    results.append({
                        "game_id": ex["game_id"],
                        "turn_idx": ex["turn_idx"],
                        "prompt": prompt,
                        "reference": reference,
                        "prediction": prediction,
                        "matched": prediction == reference,
                    })
                except Exception as e:
                    print(f"Error processing example: {e}")
                    # Add a placeholder for the failed example
                    results.append({
                        "game_id": ex.get("game_id", "unknown"),
                        "turn_idx": ex.get("turn_idx", -1),
                        "prompt": ex.get("prompt", ""),
                        "reference": ex.get("reference", ""),
                        "prediction": "ERROR",
                        "matched": False,
                    })
        
        return results
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate evaluation metrics from results."""
        references = [ex["reference"] for ex in results]
        predictions = [ex["prediction"] for ex in results]
        
        # Calculate exact match manually
        exact_matches = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
        total = len(predictions)
        
        # Calculate metrics
        metrics = {
            "exact_match": exact_matches / total if total > 0 else 0,
            "accuracy": exact_matches / total if total > 0 else 0,
            "total_examples": total,
            "correct_predictions": exact_matches,
        }
        
        return metrics
    
    def save_results(self, results: List[Dict], metrics: Dict):
        """Save evaluation results and metrics."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_path = self.output_dir / f"results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        
        # Save metrics summary
        metrics_path = self.output_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Evaluation results saved to {results_path}")
        print(f"Evaluation metrics saved to {metrics_path}")
        
        # Print metrics summary
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    def evaluate(self, num_examples: Optional[int] = None, do_sample: bool = True, temperature: float = 0.7, top_p: float = 0.9):
        """Run the full evaluation pipeline."""
        # Load dataset
        self.load_dataset()
        
        # Preprocess examples
        examples = self.preprocess_examples(num_examples)
        
        # Generate predictions
        results = self.generate_predictions(examples, do_sample=do_sample, temperature=temperature, top_p=top_p)
        
        # Calculate metrics
        metrics = self.calculate_metrics(results)
        
        # Save results and metrics
        self.save_results(results, metrics)
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate Gomoku GPT-2 model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to processed dataset")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to evaluate (None for all)")
    parser.add_argument("--do_sample", type=bool, default=True, help="Whether to use sampling")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top_p")
    
    args = parser.parse_args()
    
    evaluator = GomokuEvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
    )
    
    evaluator.evaluate(num_examples=args.num_examples, do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p)

if __name__ == "__main__":
    main()
