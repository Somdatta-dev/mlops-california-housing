/**
 * Custom hooks for WebSocket connections
 */

import { useState, useEffect, useCallback } from 'react';
import { wsClient, WebSocketEventHandlers, TrainingStatus, GPUMetrics } from '@/services/websocket';
import { SystemHealth } from '@/types';

// Hook for WebSocket connection management
export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionState, setConnectionState] = useState<string>('disconnected');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const handlers: WebSocketEventHandlers = {
      onConnect: () => {
        setIsConnected(true);
        setConnectionState('connected');
        setError(null);
      },
      onDisconnect: () => {
        setIsConnected(false);
        setConnectionState('disconnected');
      },
      onError: (error) => {
        setError(error);
      },
    };

    wsClient.setEventHandlers(handlers);

    // Attempt to connect
    wsClient.connect().catch((error) => {
      console.error('Failed to connect to WebSocket:', error);
      setError('Failed to connect');
    });

    // Update connection state periodically
    const interval = setInterval(() => {
      setConnectionState(wsClient.connectionState);
      setIsConnected(wsClient.isConnected);
    }, 1000);

    return () => {
      clearInterval(interval);
      wsClient.disconnect();
    };
  }, []);

  const reconnect = useCallback(() => {
    setError(null);
    wsClient.connect().catch((error) => {
      console.error('Failed to reconnect:', error);
      setError('Failed to reconnect');
    });
  }, []);

  const send = useCallback((type: string, data: unknown) => {
    wsClient.send(type, data);
  }, []);

  return {
    isConnected,
    connectionState,
    error,
    reconnect,
    send,
  };
}

// Hook for real-time predictions
export function useRealtimePredictions() {
  const [predictions, setPredictions] = useState<unknown[]>([]);
  const [latestPrediction, setLatestPrediction] = useState<unknown | null>(null);

  useEffect(() => {
    const handlers: WebSocketEventHandlers = {
      onPrediction: (data) => {
        setLatestPrediction(data);
        setPredictions(prev => [data, ...prev.slice(0, 99)]); // Keep last 100 predictions
      },
    };

    wsClient.setEventHandlers(handlers);
  }, []);

  const clearPredictions = useCallback(() => {
    setPredictions([]);
    setLatestPrediction(null);
  }, []);

  return {
    predictions,
    latestPrediction,
    clearPredictions,
  };
}

// Hook for training status
export function useTrainingStatus() {
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    status: 'idle',
    progress: 0,
  });

  useEffect(() => {
    const handlers: WebSocketEventHandlers = {
      onTrainingStatus: (data) => {
        setTrainingStatus(data);
      },
    };

    wsClient.setEventHandlers(handlers);
  }, []);

  const startTraining = useCallback((config: Record<string, unknown>) => {
    wsClient.send('start_training', config);
  }, []);

  const pauseTraining = useCallback(() => {
    wsClient.send('pause_training', {});
  }, []);

  const stopTraining = useCallback(() => {
    wsClient.send('stop_training', {});
  }, []);

  return {
    trainingStatus,
    startTraining,
    pauseTraining,
    stopTraining,
  };
}

// Hook for GPU metrics
export function useGPUMetrics() {
  const [gpuMetrics, setGPUMetrics] = useState<GPUMetrics | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<GPUMetrics[]>([]);

  useEffect(() => {
    const handlers: WebSocketEventHandlers = {
      onGPUMetrics: (data) => {
        setGPUMetrics(data);
        setMetricsHistory(prev => [data, ...prev.slice(0, 59)]); // Keep last 60 data points (1 minute at 1s intervals)
      },
    };

    wsClient.setEventHandlers(handlers);
  }, []);

  return {
    gpuMetrics,
    metricsHistory,
  };
}

// Hook for system health monitoring
export function useSystemHealth() {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [alerts, setAlerts] = useState<string[]>([]);

  useEffect(() => {
    const handlers: WebSocketEventHandlers = {
      onSystemHealth: (data) => {
        setSystemHealth(data);
        
        // Check for alerts
        if (data.cpu_usage > 90) {
          setAlerts(prev => [...prev, 'High CPU usage detected']);
        }
        if (data.memory_usage > 90) {
          setAlerts(prev => [...prev, 'High memory usage detected']);
        }
        if (data.gpu_temperature > 85) {
          setAlerts(prev => [...prev, 'High GPU temperature detected']);
        }
      },
      onError: (error) => {
        setAlerts(prev => [...prev, `System error: ${error}`]);
      },
    };

    wsClient.setEventHandlers(handlers);
  }, []);

  const clearAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  return {
    systemHealth,
    alerts,
    clearAlerts,
  };
}