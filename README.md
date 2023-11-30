# NeuralNet

COMP 3940 Project

## Installation (As of November 30, 2023)

1. Clone the repository.

2. Add `uploads/` folder to root directory.

3. Open `server.js` and change the `pythonExecutablePath` to your python path.
    - Windows example: `C:\\Users\\<username>\\AppData\\Local\\Programs\\Python\\Python39\\python.exe`.
      - You can find this by running `where python` in your Windows Command Prompt.
    - MacOS example: `/usr/bin/python3`.
      - You can find this by running `which python3` in your MacOS terminal.
  
4. Also in `server.js`, change the `pythonProcess` path to the `inference.py`'s path in your machine.
    - Windows example: `C:/Users/bardi/<where you cloned the repo>/Nebula/backend/inference.py`.
      - You can find this by right-clicking the file in your IDE and selecting "Copy Path".
    - MacOS example: `/Users/ean/<where you cloned the repo>/Nebula/backend/inference.py`.
      - You can find this by right-clicking the file in your IDE and selecting "Copy Path".

5. Install these in your Command Line / Terminal:
    - `npm i node`
    - `npm i nodemon`

6. Start the program with `npm start`.

7. Open `localhost:3000` in your browser.
