AI Tools Hub
A modern, unified platform providing multiple AI-powered tools with an enhanced user interface and seamless user experience.
🚀 Overview
AI Tools Hub is a comprehensive web application that combines multiple AI services into a single, intuitive platform. Built with a modern tech stack featuring a Next.js frontend and Python backend, it offers professional-grade AI tools with enterprise-level performance.
✨ Features
🤖 AI Chatbot

Powered by Google Gemini: Advanced conversational AI with natural language understanding
Context-aware responses: Maintains conversation history for better interactions
Multi-topic support: Handles various queries from technical to creative

🖼️ Advanced Photo Editor

AI-powered enhancement: Automatic image quality improvement
Smart filters: Intelligent photo processing with one-click enhancements
Batch processing: Handle multiple images simultaneously
Cloud integration: Seamless image storage and retrieval via ImageKit

🎵 Text-to-Speech Studio

Natural voice synthesis: High-quality speech generation using Azure Cognitive Services
Background music integration: Add ambient music from Freesound library
Voice customization: Multiple voice options and speech parameters
Audio export: Download generated audio in various formats

📊 User Analytics

Activity tracking: Monitor usage patterns and preferences
Performance metrics: Track tool usage and user engagement
Data visualization: Interactive charts and insights

🛠️ Tech Stack
Frontend

Next.js 14: React framework with App Router
TypeScript: Type-safe development
Tailwind CSS: Modern utility-first styling
shadcn/ui: Beautiful, accessible UI components

Backend

Python 3.8+: Core backend logic
FastAPI: High-performance API framework
MongoDB: NoSQL database for user data and analytics
Docker: Containerized deployment

📋 Prerequisites

Node.js 18.17 or later
Python 3.8 or later
MongoDB (local or cloud instance)
Docker (optional, for containerized deployment)

Required API Keys

Google Gemini API: For chatbot functionality
Azure Cognitive Services: For text-to-speech
ImageKit: For image processing and storage
Freesound: For background music library

🚀 Quick Start
1. Clone the Repository
bashgit clone https://github.com/harshagar12/ai-tools-hub.git
cd ai-tools-hub
2. Backend Setup
bash# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
3. Frontend Setup
bashcd frontend
npm install
# or
yarn install
4. Environment Configuration
bash# Copy environment template
cp .env.example .env.local

# Edit .env.local with your API keys and configuration
5. Start the Application
Development Mode
bash# Start backend (from root directory)
python api_server.py

# Start frontend (in new terminal)
cd frontend
npm run dev
Production Mode
bash# Build frontend
cd frontend
npm run build

# Start with Docker
docker-compose up -d
🔧 Configuration
Create a .env.local file in the root directory with the following variables:
env# API Keys
GOOGLE_GEMINI_API_KEY=your_gemini_key_here
AZURE_SPEECH_KEY=your_azure_key_here
AZURE_SPEECH_REGION=your_azure_region
IMAGEKIT_PUBLIC_KEY=your_imagekit_public_key
IMAGEKIT_PRIVATE_KEY=your_imagekit_private_key
FREESOUND_API_KEY=your_freesound_key_here

# Database
MONGODB_URI=mongodb://localhost:27017/ai-tools-hub

# Server Configuration
BACKEND_PORT=8000
FRONTEND_PORT=3000
🧪 Testing
bash# Run backend tests
python -m pytest tests/ -v

# Run frontend tests
cd frontend
npm test
📚 API Documentation
Once the backend is running, visit:

Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc

🐳 Docker Deployment
bash# Build and run with Docker Compose
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
🤝 Contributing
We welcome contributions! Please follow these steps:

Fork the repository
Create a feature branch: git checkout -b feature/amazing-feature
Commit your changes: git commit -m 'Add amazing feature'
Push to the branch: git push origin feature/amazing-feature
Open a Pull Request

Development Guidelines

Follow TypeScript/Python coding standards
Add tests for new features
Update documentation as needed
Ensure all tests pass before submitting

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🆘 Support

Issues: GitHub Issues
Discussions: GitHub Discussions
Email: support@aitoolshub.com

🎯 Roadmap

 Voice chat integration
 Advanced photo filters
 Multi-language support
 Mobile app development
 API rate limiting
 User authentication system

🙏 Acknowledgments

Google Gemini for powerful AI capabilities
Microsoft Azure for text-to-speech services
ImageKit for image processing infrastructure
Freesound community for audio resources