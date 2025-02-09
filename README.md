# Denoising autoencoder 
DAE для автоматичного виправлення граматичних помилок у пошукових запитах користувачів на e-commerce сайті

проєкт розроблений за допомогою PyTorch

## Інсталяція

1. Клонуйте репозиторій:
```bash
git clone https://github.com/kolenchuk/dae
cd dae

# Створіть віртуальне середовище
python -m venv pytorch_env

# Активуйте віртуальне середовище
# На Windows:
pytorch_env\Scripts\activate
# На Unix чи MacOS:
source pytorch_env/bin/activate

pip install -r requirements.txt

dae/
├── data/               # Dataset files
├── models/             # Saved models
├── reports/            # Validation reports
├── src/                # Source code
├── requirements.txt    # Project dependencies
├── app.py              # Скрипт запуску апі
├── Dockerfile          #
├── docker-compose.yaml #
└── README.md


This README.md includes:
- Clear installation instructions
- Project structure
- Requirements
