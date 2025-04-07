// server.js
import express from 'express';
import Redis from 'ioredis';
import { v4 as uuidv4 } from 'uuid';
import bodyParser from 'body-parser';
import dotenv from "dotenv";

dotenv.config();

// Create Express app
const app = express();
const port = process.env.PORT || 3000;

// Redis connection
const redis = new Redis({
  host: process.env.REDIS_HOST || 'localhost',
  port: process.env.REDIS_PORT || 6379,
  password: process.env.REDIS_PASSWORD || ''
});

// Queue name
const QUEUE_NAME = 'sdxl_jobs';

// Middleware
app.use(bodyParser.json());

// Routes
app.post('/v6/images/text2img', async (req, res) => {
  try {
    // Generate a job ID
    const jobId = uuidv4();
    
    // Validate request
    const { prompt } = req.body;
    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }
    
    // Create job payload
    const jobData = {
      id: jobId,
      status: 'processing',
      createdAt: Date.now(),
      ...req.body
    };
    
    // Store job data in Redis with expiration (24 hours)
    await redis.setex(`job:${jobId}`, 86400, JSON.stringify(jobData));
    
    // Add job to queue
    await redis.rpush(QUEUE_NAME, jobId);
    console.log(`Job ${jobId} added to queue`);
    
    // Return job ID to client
    res.status(202).json({
      jobId,
      status: 'processing',
      fetch_result: `https://${process.env.RUNPOD_POD_ID}-3000.proxy.runpod.net/status/${jobId}`,
      message: 'Job added to queue'
    });
  } catch (error) {
    console.error('Error adding job to queue:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

// Get job status
app.post('/status/:jobId', async (req, res) => {
  try {
    const { jobId } = req.params;
    
    // Get job data from Redis
    const jobData = await redis.get(`job:${jobId}`);
    
    if (!jobData) {
      return res.status(404).json({ error: 'Job not found' });
    }
    
    // Parse job data
    const job = JSON.parse(jobData);
    
    // Return job status to client
    res.status(200).json({
      jobId,
      status: job.status,
      result: job.result,
      createdAt: job.createdAt,
      completedAt: job.completedAt
    });
  } catch (error) {
    console.error('Error getting job status:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

// Start server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});

// Handle Redis connection errors
redis.on('error', (error) => {
  console.error('Redis connection error:', error);
});

// Handle shutdown
process.on('SIGTERM', async () => {
  console.log('Shutting down server...');
  await redis.quit();
  process.exit(0);
});