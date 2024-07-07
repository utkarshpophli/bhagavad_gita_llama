# 🕉️ Gita Guru: Your AI Guide to the Bhagavad Gita 🧘‍♂️

Welcome to Gita Guru, where ancient wisdom meets cutting-edge AI! 🚀

## 🌟 What is Gita Guru?

Gita Guru is not just another chatbot. It's your personal spiritual companion, powered by the wisdom of the Bhagavad Gita and the intelligence of LLaMA-2. We've fine-tuned one of the most advanced language models to become an expert in the teachings of this ancient Indian scripture.

## ✨ Features

- 🧠 Fine-tuned LLaMA-2-7b-chat model, now a Gita genius!
- 💬 Interactive Streamlit app for enlightening conversations
- 📊 TensorBoard integration for those who love to see the learning in action
- 🚀 Optimized with QLoRA for efficient training and deployment

## 🛠️ Installation

1. Clone this cosmic repository:
```
git clone https://github.com/utkarshpophli/bhagavad_gita_llama.git
cd gita-guru
```
2. Create a virtual environment 
```
conda create -n venv
conda activate venv
```

3. Install the divine dependencies:
```
pip install -r requirements.txt
```

4. Align your chakras with Hugging Face:
```
huggingface-cli login
```
## 🧘‍♀️ Usage

### Training the Guru

To imbue our AI with the wisdom of the Gita:
```
python src/train.py
```

Watch as the model transcends its initial knowledge!

### Seeking Guidance

To start your spiritual journey with Gita Guru:
```
streamlit run app.py
```
Open your browser and navigate to the provided URL. Ask your questions and receive insights drawn from the depths of the Bhagavad Gita!

## 🧪 Experiment and Explore

Dive deeper into the data and model behavior:

1. Check out `notebooks/exploration.ipynb` for dataset insights.
2. Use TensorBoard to visualize the training process:
```
tensorboard --logdir results/runs
```

## 🤝 Contributing

We welcome contributions from all seekers of knowledge and truth! Whether you're a ML guru, a Gita scholar, or just enthusiastic about the project, feel free to:

- 🐛 Report bugs
- 💡 Suggest enhancements
- 🖊️ Improve documentation
- 🧑‍💻 Submit pull requests

Check out our [Contribution Guidelines](CONTRIBUTING.md) for more details.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- The creators of LLaMA-2 for the incredible base model
- The Hugging Face team for their transformers library
- The countless teachers and scholars who have preserved and interpreted the Bhagavad Gita over millennia

---

Remember, just as Krishna guided Arjuna, let Gita Guru guide you through the battlefield of life! May your code be bug-free and your mind be at peace. 🕊️