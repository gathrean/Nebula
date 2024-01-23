const express = require('express');
const multer = require('multer');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const port = 3000;

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    },
});

const upload = multer({ storage });

app.use(express.static(path.join(__dirname, 'frontend')));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.post('/upload', upload.single('audio'), (req, res) => {
    const audioFilePath = path.join('uploads', req.file.filename);
    const pythonExecutablePath = '/Library/Frameworks/Python.framework/Versions/3.11/bin/python3';

    const pythonProcess = spawn(pythonExecutablePath, ['/Users/ravdeepaulakh/Downloads/Nebula/backend/inference.py', audioFilePath]);

    let predictedLabels = [];
    let errorOccurred = false;

    pythonProcess.stdout.on('data', (data) => {
        const output = data.toString().trim();
        console.log(`Received from Python script (stdout): ${output}`);

        // Assuming the output contains labels, parse and store them
        const lines = output.split('\n');
        predictedLabels = lines.map(line => line.trim());
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Error from Python script (stderr): ${data}`);
        errorOccurred = true;
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python script exited with code ${code}`);
        console.log(`Final predicted labels: ${predictedLabels}`);

        if (errorOccurred) {
            res.status(500).json({ success: false, error: 'Internal Server Error' });
        } else {
            console.log("sending json")
            res.status(200).json({ success: true, predicted: predictedLabels });
        }
    });
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
