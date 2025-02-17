Hello, this is a simple app to recognize fruits using machine learning model with PyTorch.

How to use in VS Code:

1. Clone this repository to your desired location

2. Open project and create in terminal python virtual environment
```bash
python -m venv env
env\Scripts\activate
```

3. Install dependencies
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn GPUtil
```

4. The structure looks like this:

mlfruits/
├── data/
│   ├── train/
│   │   ├── apple/
│   │   ├── banana/
│   │   └── orange/
│   └── validation/
│       ├── apple/
│       ├── banana/
│       └── orange/
├── env/Scripts/
│   ├── analyze_dataset.py
│   ├── load_data.py
│   ├── model.py
│   ├── train.py
│   └── validate.py
└── README.md

Create a DATA folder and put inside the TRAIN and VALIDATION folders, each containing a separated class of fruit fileld with pictures.jpg of your choice.
YOU MUST keep the classes symetrical so for each class in train folder, a corresponding class should be in validation folder.

I recommend using train/validation pictures count ratio around 80/20 or 90/10. For example 40 train appled and 8 validation apples.

5. You can analyze your prepared dataset distribution:
```bash
python env/Scripts/analyze_dataset.py
```
This will show you a graph of pictures distribution in your class folders.

6. Train the model:
```bash
python env/Scripts/train.py
```

7. Validate results:
```bash
python env/Scripts/validate.py
```


========================
Additional informations:
========================

1. model.py uses an imported from torchvision ResNet18 model that is tailored to support well pictures like fruits. You can replace it with any other model your want, if you decide to train a different kind of pictures or to use a default CNN model. Modify then accordingly class FruitClassifier(nn.Module) and update references to it in other files.

2. load_data.py transforms initial picutres and standarize them. You can use any size and resolution you want. The program will scale them down to the popular 224x224 res and transform without any distortions. For pictures that are not a square, it will scale it down along with the longer side and add padding to shorter side.
Here you can also play with transform settings such as rotation, color variances etc. Increasing their ranges may train the model better.

3. train.py containa few most important variables listed on top for easy access: epochs, learning_rate, max_learning_rate, weight_decay. You can play with them to refine the training process.
It has a function to stop the process early if it detects no further improvement in accuracy after next 10 epochs (you can still adjust this value by your wish, search for 'patience') variable.
Trained model is saved with a timestamp after the process is done, however the program saves also best_model.pth with the highest achieved accuracy during the process. So that is why in validate.py I use best_model as reference. You can update your 'models' folder path (serach for MODELS_DIR).

4. validate.py is to show plots with statistics of how accurate the model recognized pictures from validation classes.
