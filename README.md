
# Neural Network from Scratch

Created complete Neural Network model from scratch in python with different type of activation functions to select from like ReLu, Sigmoid and with different variants of gradient decent like adam optimizer, gd with momentum etc.
## Features

- Uses numpy matrix calculation to make it fast
- Supports 3 types of activation functions
    - ReLu
    - Sigmoid
    - Tanh
- Supports 3 varients of gradient decent
    - Vanilla GD
    - GD with momentum
    - Adam optimizer
- Funtion to plot loss
- Custom number of layers
- RMSE type parameters to test accuracy
- Used matrix arithmetic to make program fast
- Train test split
## Tech Stack

**Python**: The core language used to create the model.

**Numpy**: To handle arrays

**Matplotlib**: For visualizing plots
## Run Locally

Clone the project

```bash
git clone https://github.com/menishantsharma/Neural-Network-from-Scratch
```

Go to the project directory

```bash
cd Neural-Network-from-Scratch
```

Create python virtual environment

```bash
python3 -m venv myvenv
```

Activate the environment
```bash
source myenv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

You can set the parameters like which gradient decent varient to use etc in nn_template.py file itself.

Run program

```bash
python3 nn_template.py
```
## Authors

- **Nishant Sharma** - MTech CSE Student at IIT Bombay
- **Github** - [@Nishant Sharma](https://github.com/menishantsharma)


## License

[MIT](https://choosealicense.com/licenses/mit/)

