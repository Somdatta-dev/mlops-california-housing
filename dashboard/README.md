# MLOps Dashboard

A comprehensive Next.js dashboard for the MLOps California Housing prediction platform with real-time monitoring, GPU metrics, and training management.

## Features

- **Real-time Dashboard**: Live system status, model performance, and GPU metrics
- **Prediction Interface**: Interactive forms for single and batch predictions
- **Training Management**: Start, pause, and monitor model training with real-time progress
- **GPU Monitoring**: Real-time GPU utilization, memory usage, and temperature tracking
- **Data Explorer**: Browse prediction history with filtering and export capabilities
- **System Health**: Monitor API health, resource usage, and system alerts
- **WebSocket Integration**: Real-time updates for all dashboard components

## Technology Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: TailwindCSS with shadcn/ui components
- **Charts**: Recharts for data visualization
- **Real-time**: WebSocket client for live updates
- **Icons**: Lucide React icons
- **Date Handling**: date-fns for date formatting

## Project Structure

```
src/
├── app/                    # Next.js app router pages
├── components/
│   ├── dashboard/         # Dashboard-specific components
│   ├── layout/           # Layout components (Header, Sidebar, Layout)
│   └── ui/               # shadcn/ui components
├── hooks/                # Custom React hooks
│   ├── useApi.ts         # API interaction hooks
│   └── useWebSocket.ts   # WebSocket connection hooks
├── services/             # External service clients
│   ├── api.ts           # FastAPI client
│   └── websocket.ts     # WebSocket client
├── types/               # TypeScript type definitions
├── utils/               # Utility functions
└── lib/                 # Library configurations
```

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Running FastAPI backend (see main project README)

### Installation

1. Install dependencies:
```bash
npm install
```

2. Set up environment variables:
```bash
cp .env.local.example .env.local
# Edit .env.local with your API endpoints
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

### Environment Variables

- `NEXT_PUBLIC_API_URL`: FastAPI backend URL (default: http://localhost:8000)
- `NEXT_PUBLIC_WS_URL`: WebSocket endpoint URL (default: ws://localhost:8000/ws)

## Available Scripts

- `npm run dev`: Start development server
- `npm run build`: Build for production
- `npm run start`: Start production server
- `npm run lint`: Run ESLint
- `npm run type-check`: Run TypeScript type checking

## API Integration

The dashboard integrates with the FastAPI backend through:

### REST API Endpoints
- `GET /health`: System health status
- `GET /model/info`: Current model information
- `POST /predict`: Single prediction
- `POST /predict/batch`: Batch predictions
- `GET /predictions`: Prediction history

### WebSocket Events
- `prediction`: Real-time prediction updates
- `training_status`: Training progress and status
- `gpu_metrics`: GPU utilization and metrics
- `system_health`: System resource monitoring

## Components Overview

### Layout Components
- **Header**: System status, notifications, and user controls
- **Sidebar**: Navigation menu with collapsible design
- **Layout**: Main layout wrapper with responsive design

### Dashboard Components
- **DashboardOverview**: Main dashboard with key metrics and status cards
- **PredictionDashboard**: Real-time prediction interface and history
- **TrainingInterface**: Model training controls and progress monitoring
- **DatabaseExplorer**: Prediction history browser with filtering
- **SystemMonitor**: System health and resource monitoring

### Custom Hooks
- **useApi**: Generic API interaction with loading states
- **useWebSocket**: WebSocket connection management
- **useRealtimePredictions**: Real-time prediction feed
- **useTrainingStatus**: Training status monitoring
- **useGPUMetrics**: GPU metrics tracking

## Styling and Theming

The dashboard uses TailwindCSS with shadcn/ui components for consistent styling:

- **Color Scheme**: Professional blue and gray palette
- **Typography**: Inter font for clean readability
- **Components**: Consistent shadcn/ui component library
- **Responsive**: Mobile-first responsive design
- **Dark Mode**: Ready for dark mode implementation

## Real-time Features

### WebSocket Integration
- Automatic reconnection with exponential backoff
- Event-based message handling
- Connection state management
- Error handling and recovery

### Live Updates
- Real-time prediction feed
- GPU metrics streaming
- Training progress updates
- System health monitoring
- Alert notifications

## Development Guidelines

### Code Organization
- Components are organized by feature/domain
- Hooks are separated by functionality
- Types are centralized in `/types`
- Utilities are shared across components

### State Management
- React hooks for local state
- Custom hooks for shared logic
- WebSocket state management
- API state with loading/error handling

### Error Handling
- Graceful API error handling
- WebSocket connection recovery
- User-friendly error messages
- Fallback UI states

## Deployment

### Production Build
```bash
npm run build
npm run start
```

### Docker Deployment
```bash
docker build -t mlops-dashboard .
docker run -p 3000:3000 mlops-dashboard
```

### Environment Configuration
- Set production API URLs
- Configure WebSocket endpoints
- Enable production optimizations
- Set up monitoring and logging

## Contributing

1. Follow TypeScript best practices
2. Use shadcn/ui components when possible
3. Implement proper error handling
4. Add loading states for async operations
5. Ensure responsive design
6. Write meaningful component documentation

## License

This project is part of the MLOps California Housing prediction platform.