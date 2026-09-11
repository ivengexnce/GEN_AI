"""Unified CLI Entrypoint for AI Model and Agentic Framework."""
import argparse
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.core.config import load_config
from src.model.trainer import ModelTrainer
from src.model.predictor import SentimentPredictor
from src.agent.react_agent import ReActAgent



def run_training(config):
    """Train the PyTorch neural network."""
    print("\n=======================================================")
    print("  🚀 INITIATING AI MODEL TRAINING (PyTorch)")
    print("=======================================================\n")
    trainer = ModelTrainer(config)
    results = trainer.train()
    print("\nTraining Metrics:", results)
    print(f"Model saved to: {config.training.model_save_dir}/{config.training.model_save_name}")


def run_prediction(text: str, config):
    """Run model inference on single text."""
    try:
        predictor = SentimentPredictor(config)
        result = predictor.predict(text)
        print("\n=======================================================")
        print("  🎯 AI MODEL INFERENCE RESULT")
        print("=======================================================")
        print(f"Input Text:    \"{result['text']}\"")
        print(f"Prediction:    {result['label']}")
        print(f"Confidence:    {result['confidence'] * 100:.2f}%")
        print(f"Distribution:  {result['probabilities']}")
        print("=======================================================\n")
    except FileNotFoundError as e:
        print(f"\n[!] Error: {e}")
        print("[!] Hint: Run `python main.py train` to train and save the model first.\n")


def run_agent(goal: str, config):
    """Execute the autonomous ReAct agent."""
    agent = ReActAgent(config)
    agent.run(goal)


def run_interactive(config):
    """Interactive console to test models and agents dynamically."""
    print("\n=======================================================")
    print("  🧠 WELCOME TO THE GENESIS AI & AGENT WORKBENCH")
    print("=======================================================")
    print("Commands:")
    print("  1. 'train'               - Train the PyTorch model")
    print("  2. 'predict <text>'      - Predict sentiment using trained model")
    print("  3. 'agent <goal>'        - Run autonomous agent on a goal")
    print("  4. 'tools'               - List registered agent tools")
    print("  5. 'exit' or 'quit'      - Exit interactive mode\n")

    agent = ReActAgent(config)

    while True:
        try:
            user_input = input("AI-Workbench> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting. Happy AI engineering!")
                break
            elif user_input.lower() == "train":
                run_training(config)
            elif user_input.lower().startswith("predict "):
                text = user_input[8:].strip()
                run_prediction(text, config)
            elif user_input.lower().startswith("agent "):
                goal = user_input[6:].strip()
                agent.run(goal)
            elif user_input.lower() == "tools":
                print("\nRegistered Tools:")
                print(agent.tools.format_tool_descriptions())
                print()
            else:
                # Default to running as agent goal
                print(f"Executing as agent goal: '{user_input}'")
                agent.run(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Genesis AI Framework: End-to-End AI Model & Agent System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: train
    subparsers.add_parser("train", help="Train the PyTorch neural network model")

    # Subcommand: predict
    predict_parser = subparsers.add_parser("predict", help="Predict sentiment of text")
    predict_parser.add_argument("--text", type=str, required=True, help="Text to classify")

    # Subcommand: agent
    agent_parser = subparsers.add_parser("agent", help="Run the autonomous ReAct agent")
    agent_parser.add_argument("--goal", type=str, required=True, help="Goal or prompt for the agent")

    # Subcommand: interactive
    subparsers.add_parser("interactive", help="Start interactive workbench session")

    args = parser.parse_args()
    config = load_config()

    if args.command == "train":
        run_training(config)
    elif args.command == "predict":
        run_prediction(args.text, config)
    elif args.command == "agent":
        run_agent(args.goal, config)
    elif args.command == "interactive" or len(sys.argv) == 1:
        run_interactive(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
