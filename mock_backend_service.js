/**
 * This file provides a simple mock backend service for development and testing
 * when the actual backend services are unavailable or unstable.
 * 
 * Run with: node mock_backend_service.js
 */

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');

const app = express();
const port = 3001;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Configure storage
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'mock_uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}_${file.originalname}`);
  },
});

const upload = multer({ storage });

// Store jobs in memory (would be a database in production)
const jobs = new Map();

// Upload endpoint
app.post('/upload', upload.single('file'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const jobId = uuidv4();
    
    jobs.set(jobId, {
      id: jobId,
      filename: req.file.originalname,
      path: req.file.path,
      status: 'processing',
      created: new Date(),
    });

    // Simulate processing completion after a delay
    setTimeout(() => {
      const job = jobs.get(jobId);
      if (job) {
        jobs.set(jobId, {
          ...job,
          status: 'completed',
          results: {
            segmentation_path: `/visualization/${jobId}`,
            volume_dimensions: [128, 128, 128],
            metastasis_count: Math.floor(Math.random() * 5) + 1,
            metastasis_volume: Math.floor(Math.random() * 1000) / 10,
          }
        });
        console.log(`Job ${jobId} completed`);
      }
    }, 15000); // Complete after 15 seconds

    return res.status(200).json({
      message: 'File uploaded successfully',
      job_id: jobId,
      status: 'processing',
    });
  } catch (error) {
    console.error('Upload error:', error);
    return res.status(500).json({ error: 'Upload failed' });
  }
});

// Results endpoint
app.get('/results/:jobId', (req, res) => {
  const jobId = req.params.jobId;
  const job = jobs.get(jobId);

  if (!job) {
    return res.status(200).json({ status: 'not_found' });
  }

  return res.status(200).json({
    status: job.status,
    ...(job.results || {}),
    job_id: jobId,
  });
});

// Status endpoint
app.get('/status/:jobId', (req, res) => {
  const jobId = req.params.jobId;
  const job = jobs.get(jobId);

  if (!job) {
    return res.status(200).json({ status: 'not_found' });
  }

  return res.status(200).json({ status: job.status });
});

// Visualization endpoint
app.get('/visualization/:jobId', (req, res) => {
  const jobId = req.params.jobId;
  const job = jobs.get(jobId);

  if (!job || job.status !== 'completed') {
    return res.status(404).json({ error: 'Visualization not available' });
  }

  // Send sample image file or create one
  const sampleImagePath = path.join(__dirname, 'mock_images', 'sample_slice.png');
  
  if (!fs.existsSync(path.dirname(sampleImagePath))) {
    fs.mkdirSync(path.dirname(sampleImagePath), { recursive: true });
  }

  if (!fs.existsSync(sampleImagePath)) {
    // Create a simple HTML with a message saying it's a mock image
    const html = `
      <html>
        <body style="display:flex;align-items:center;justify-content:center;height:100vh;background:#f0f0f0;">
          <div style="background:white;padding:20px;border-radius:8px;text-align:center;">
            <h2>Mock Visualization</h2>
            <p>Job ID: ${jobId}</p>
            <p>Status: ${job.status}</p>
          </div>
        </body>
      </html>
    `;
    fs.writeFileSync(sampleImagePath.replace('.png', '.html'), html);
    return res.sendFile(sampleImagePath.replace('.png', '.html'));
  }

  return res.sendFile(sampleImagePath);
});

// Volume dimensions endpoint
app.get('/api/volume-dimensions/:jobId', (req, res) => {
  const jobId = req.params.jobId;
  const job = jobs.get(jobId);

  if (!job || job.status !== 'completed') {
    return res.status(404).json({ error: 'Volume dimensions not available' });
  }

  return res.status(200).json({
    dimensions: [128, 128, 128]
  });
});

app.listen(port, () => {
  console.log(`Mock backend server running at http://localhost:${port}`);
  console.log('Available endpoints:');
  console.log('  POST /upload');
  console.log('  GET /results/:jobId');
  console.log('  GET /status/:jobId');
  console.log('  GET /visualization/:jobId');
  console.log('  GET /api/volume-dimensions/:jobId');
});
