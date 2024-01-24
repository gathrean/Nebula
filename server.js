import express from 'express';
import multer from 'multer';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fetch from 'node-fetch';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;

// Update the ngrok URL here
const flaskServerURL = 'https://d187-2604-3d08-7077-d200-38e9-945c-5f3e-489d.ngrok-free.app/execute-python';

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    },
});

const upload = multer({ storage });

app.use(express.static(join(__dirname, 'frontend')));

app.get('/', (req, res) => {
    res.sendFile(join(__dirname, 'index.html'));
});

app.post('/upload', upload.single('audio'), async (req, res) => {
    const audioFilePath = join(__dirname, 'uploads', req.file.filename);

    try {
        const response = await fetch(flaskServerURL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ audioFilePath }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();

        // Log the response from Flask for debugging
        console.log('Flask Response:', data);

        // Check if 'predicted' field exists in the response
        if ('predicted' in data) {
            res.status(200).json({ success: true, predicted: data.predicted });
        } else {
            res.status(500).json({ success: false, error: 'Malformed response from Flask' });
        }
    } catch (error) {
        console.error(error);
        res.status(500).json({ success: false, error: 'Internal Server Error' });
    }
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
