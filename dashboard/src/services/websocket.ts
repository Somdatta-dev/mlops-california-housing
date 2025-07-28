/**
 * WebSocket client for real-time communication with FastAPI backend
 */

export interface WebSocketMessage {
  type: string;
  data: unknown;
  timestamp: string;
}

export interface GPUMetrics {
  utilization: number;
  memory_used: number;
  memory_total: number;
  temperature: number;
  power_usage: number;
}

export interface TrainingStatus {
  status: 'idle' | 'training' | 'paused' | 'completed' | 'error';
  progress: number;
  current_epoch?: number;
  total_epochs?: number;
  current_loss?: number;
  model_type?: string;
  gpu_metrics?: GPUMetrics;
}

export interface SystemHealth {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  gpu_temperature: number;
  api_response_time: number;
  active_connections: number;
}

export type WebSocketEventType = 
  | 'prediction'
  | 'training_status'
  | 'gpu_metrics'
  | 'system_health'
  | 'error';

export interface WebSocketEventHandlers {
  onPrediction?: (data: unknown) => void;
  onTrainingStatus?: (data: TrainingStatus) => void;
  onGPUMetrics?: (data: GPUMetrics) => void;
  onSystemHealth?: (data: SystemHealth) => void;
  onError?: (error: string) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 1000;
  private handlers: WebSocketEventHandlers = {};
  private isConnecting = false;

  constructor(url: string = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws') {
    this.url = url;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      if (this.isConnecting) {
        reject(new Error('Connection already in progress'));
        return;
      }

      this.isConnecting = true;

      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.handlers.onConnect?.();
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
            this.handlers.onError?.('Failed to parse message');
          }
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket disconnected:', event.code, event.reason);
          this.isConnecting = false;
          this.handlers.onDisconnect?.();
          
          if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.isConnecting = false;
          this.handlers.onError?.('Connection error');
          reject(error);
        };
      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = this.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      this.connect().catch((error) => {
        console.error('Reconnection failed:', error);
      });
    }, delay);
  }

  private handleMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case 'prediction':
        this.handlers.onPrediction?.(message.data);
        break;
      case 'training_status':
        this.handlers.onTrainingStatus?.(message.data as TrainingStatus);
        break;
      case 'gpu_metrics':
        this.handlers.onGPUMetrics?.(message.data as GPUMetrics);
        break;
      case 'system_health':
        this.handlers.onSystemHealth?.(message.data as SystemHealth);
        break;
      case 'error':
        this.handlers.onError?.(message.data as string);
        break;
      default:
        console.warn('Unknown message type:', message.type);
    }
  }

  send(type: string, data: unknown): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const message: WebSocketMessage = {
        type,
        data,
        timestamp: new Date().toISOString(),
      };
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }

  setEventHandlers(handlers: WebSocketEventHandlers): void {
    this.handlers = { ...this.handlers, ...handlers };
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  get connectionState(): string {
    if (!this.ws) return 'disconnected';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      case WebSocket.CLOSING:
        return 'closing';
      case WebSocket.CLOSED:
        return 'disconnected';
      default:
        return 'unknown';
    }
  }
}

// Export singleton instance
export const wsClient = new WebSocketClient();
export default WebSocketClient;