const express = require("express");
const dotenv = require("dotenv");
dotenv.config();
const app = express();
const port = process.env.PORT || 3000;
const TestController = require("./controller/controller.js");
const axios = require("axios");
const pdfParse = require("pdf-parse");
const fs = require("fs");
const multer = require("multer");
const { spawn } = require("child_process");
const path = require("path");
app.use(express.json());

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/");
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, Date.now() + ext);
  },
});

const upload = multer({ storage });

app.get("/train-model", (req, res) => {
  let responseSent = false; // Track whether a response has already been sent

  const pythonProcess = spawn("python", ["training_data.py"]);

  pythonProcess.stdout.on("data", (data) => {
    console.log(`stdout: ${data}`);
    if (!responseSent) {
      res.send({ message: `Data training is in progress` });
      responseSent = true; // Mark that a response has been sent
    } else {
      console.log("Attempted to send response again from stdout");
    }
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error(`stderr: ${data}`);
    if (!responseSent) {
      res.status(500).send(`Error: ${data}`);
      responseSent = true; // Mark that a response has been sent
    } else {
      console.log("Attempted to send response again from stderr");
    }
  });

  pythonProcess.on("close", (code) => {
    console.log(`child process exited with code ${code}`);
    if (!responseSent) {
      res.send(`Process closed with code ${code}`);
      responseSent = true; // Mark that a response has been sent
    } else {
      console.log("Attempted to send response again from close");
    }
  });
});

app.post("/process-cv", upload.single("cv"), async (req, res) => {
  try {
    // Read the file buffer
    const dataBuffer = fs.readFileSync(req.file.path);

    // Parse the PDF
    const data = await pdfParse(dataBuffer);
    const text = data.text;

    // Send CV text to the Python backend
    const response = await axios.post("http://localhost:5000/parse-cv", {
      text: text,
    });

    // Return the parsed data to the client
    res.json(response.data);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Error parsing CV" });
  }
});
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
