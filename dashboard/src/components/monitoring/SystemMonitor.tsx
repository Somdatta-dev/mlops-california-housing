/**
 * SystemMonitor Component
 * 
 * Provides real-time system monitoring with live API health status,
 * resource monitoring, and performance visualization.
 */

'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  Server, 
  Cpu, 
  HardDrive, 
  Zap, 
  Thermometer,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Minus,
  Wifi,
  WifiOff
} from 'lucide-react';
import { useWebSocket, useSystemHealth, useGPUMetrics } from '@/hooks/useWebSocket';
import { useApi } from '@/hooks/useApi';
import { apiClient } from '@/services/api';
import type { 
  SystemMetrics, 
  APIHealthMetrics, 
  GPUMetrics 
} from '@/types';

interface SystemMonitorProps {
  className?: string;
  showHeader?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export function SystemMonitor({ 
  className = '', 
  showHeader = true, 
  autoRefresh = true,
  refreshInterval = 5000 
}: SystemMonitorProps) {
  const { isConnected } = useWebSocket();
  const { systemHealth } = useSystemHealth();
  const { gpuMetrics } = useGPUMetrics();
  
  // State for API-fetched data
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [apiHealth, setApiHealth] = useState<APIHealthMetrics | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Fetch system data
  const fetchSystemData = async () => {
    setIsLoading(true);
    try {
      const [systemData, apiData] = await Promise.all([
        apiClient.getSystemMetrics(),
        apiClient.getAPIHealthMetrics()
      ]);
      
      setSystemMetrics(systemData);
      setApiHealth(apiData);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to fetch system data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-refresh effect
  useEffect(() => {
    fetchSystemData();

    if (autoRefresh) {
      const interval = setInterval(fetchSystemData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  // Helper functions
  const formatBytes = (bytes: number) => {
    const gb = bytes / (1024 * 1024 * 1024);
    return `${gb.toFixed(1)} GB`;
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600';
      case 'degraded': return 'text-yellow-600';
      case 'unhealthy': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'degraded': return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      case 'unhealthy': return <XCircle className="h-4 w-4 text-red-600" />;
      default: return <Minus className="h-4 w-4 text-gray-600" />;
    }
  };

  const getProgressColor = (value: number, thresholds: { warning: number; critical: number }) => {
    if (value >= thresholds.critical) return 'bg-red-500';
    if (value >= thresholds.warning) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const getTrendIcon = (current: number, previous: number) => {
    if (current > previous) return <TrendingUp className="h-4 w-4 text-red-500" />;
    if (current < previous) return <TrendingDown className="h-4 w-4 text-green-500" />;
    return <Minus className="h-4 w-4 text-gray-500" />;
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      {showHeader && (
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">System Monitor</h2>
            <p className="text-gray-600">
              Real-time system health and performance metrics
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2 text-sm">
              {isConnected ? (
                <>
                  <Wifi className="h-4 w-4 text-green-600" />
                  <span className="text-green-600">Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="h-4 w-4 text-red-600" />
                  <span className="text-red-600">Disconnected</span>
                </>
              )}
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={fetchSystemData}
              disabled={isLoading}
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </div>
      )}

      {/* Status Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* API Status */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Server className="h-4 w-4" />
              API Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className={`text-lg font-bold ${getStatusColor(apiHealth?.status || 'unknown')}`}>
                  {apiHealth?.status?.toUpperCase() || 'UNKNOWN'}
                </span>
                {getStatusIcon(apiHealth?.status || 'unknown')}
              </div>
              {apiHealth && (
                <>
                  <div className="text-xs text-gray-500">
                    {apiHealth.response_time_ms.toFixed(1)}ms avg response
                  </div>
                  <div className="text-xs text-gray-500">
                    Uptime: {formatUptime(apiHealth.uptime_seconds)}
                  </div>
                </>
              )}
            </div>
          </CardContent>
        </Card>

        {/* CPU Usage */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Cpu className="h-4 w-4" />
              CPU Usage
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-lg font-bold">
                  {systemMetrics?.cpu_usage?.toFixed(1) || systemHealth?.cpu_usage?.toFixed(1) || '0'}%
                </span>
                {systemMetrics && getTrendIcon(systemMetrics.cpu_usage, 50)}
              </div>
              <Progress 
                value={systemMetrics?.cpu_usage || systemHealth?.cpu_usage || 0} 
                className="h-2"
                style={{
                  '--progress-background': getProgressColor(
                    systemMetrics?.cpu_usage || systemHealth?.cpu_usage || 0, 
                    { warning: 70, critical: 90 }
                  )
                } as React.CSSProperties}
              />
              {systemMetrics?.load_average && (
                <div className="text-xs text-gray-500">
                  Load: {systemMetrics.load_average[0].toFixed(2)}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Memory Usage */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <HardDrive className="h-4 w-4" />
              Memory Usage
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-lg font-bold">
                  {systemMetrics 
                    ? ((systemMetrics.memory_usage / systemMetrics.memory_total) * 100).toFixed(1)
                    : systemHealth?.memory_usage?.toFixed(1) || '0'
                  }%
                </span>
                {systemMetrics && getTrendIcon(
                  (systemMetrics.memory_usage / systemMetrics.memory_total) * 100, 
                  50
                )}
              </div>
              <Progress 
                value={
                  systemMetrics 
                    ? (systemMetrics.memory_usage / systemMetrics.memory_total) * 100
                    : systemHealth?.memory_usage || 0
                } 
                className="h-2"
                style={{
                  '--progress-background': getProgressColor(
                    systemMetrics 
                      ? (systemMetrics.memory_usage / systemMetrics.memory_total) * 100
                      : systemHealth?.memory_usage || 0, 
                    { warning: 80, critical: 95 }
                  )
                } as React.CSSProperties}
              />
              {systemMetrics && (
                <div className="text-xs text-gray-500">
                  {formatBytes(systemMetrics.memory_usage)} / {formatBytes(systemMetrics.memory_total)}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* GPU Status */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Zap className="h-4 w-4" />
              GPU Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            {gpuMetrics ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-lg font-bold">{gpuMetrics.utilization}%</span>
                  <CheckCircle className="h-4 w-4 text-green-600" />
                </div>
                <Progress 
                  value={gpuMetrics.utilization} 
                  className="h-2"
                  style={{
                    '--progress-background': getProgressColor(
                      gpuMetrics.utilization, 
                      { warning: 80, critical: 95 }
                    )
                  } as React.CSSProperties}
                />
                <div className="text-xs text-gray-500 flex justify-between">
                  <span>{gpuMetrics.temperature}°C</span>
                  <span>{Math.round(gpuMetrics.power_usage / 1000)}W</span>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-between">
                <span className="text-lg font-bold text-gray-500">N/A</span>
                <XCircle className="h-4 w-4 text-gray-400" />
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Detailed System Metrics */}
      {systemMetrics && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* System Resources */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                System Resources
              </CardTitle>
              <CardDescription>
                Detailed system resource utilization
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* CPU Details */}
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="font-medium">CPU Usage</span>
                  <span>{systemMetrics.cpu_usage.toFixed(1)}%</span>
                </div>
                <Progress value={systemMetrics.cpu_usage} className="h-2 mb-2" />
                <div className="text-xs text-gray-600">
                  Load Average: {systemMetrics.load_average.map(load => load.toFixed(2)).join(', ')}
                </div>
              </div>

              {/* Memory Details */}
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="font-medium">Memory Usage</span>
                  <span>{((systemMetrics.memory_usage / systemMetrics.memory_total) * 100).toFixed(1)}%</span>
                </div>
                <Progress value={(systemMetrics.memory_usage / systemMetrics.memory_total) * 100} className="h-2 mb-2" />
                <div className="text-xs text-gray-600">
                  {formatBytes(systemMetrics.memory_usage)} / {formatBytes(systemMetrics.memory_total)} used
                </div>
              </div>

              {/* Disk Usage */}
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="font-medium">Disk Usage</span>
                  <span>{((systemMetrics.disk_usage / systemMetrics.disk_total) * 100).toFixed(1)}%</span>
                </div>
                <Progress value={(systemMetrics.disk_usage / systemMetrics.disk_total) * 100} className="h-2 mb-2" />
                <div className="text-xs text-gray-600">
                  {formatBytes(systemMetrics.disk_usage)} / {formatBytes(systemMetrics.disk_total)} used
                </div>
              </div>
            </CardContent>
          </Card>

          {/* API Performance */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="h-5 w-5" />
                API Performance
              </CardTitle>
              <CardDescription>
                API health and performance metrics
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {apiHealth ? (
                <>
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Status</span>
                    <div className="flex items-center gap-2">
                      {getStatusIcon(apiHealth.status)}
                      <span className={getStatusColor(apiHealth.status)}>
                        {apiHealth.status.toUpperCase()}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <div className="text-gray-600">Response Time</div>
                      <div className="font-bold">{apiHealth.response_time_ms.toFixed(1)}ms</div>
                    </div>
                    <div>
                      <div className="text-gray-600">Requests/sec</div>
                      <div className="font-bold">{apiHealth.requests_per_second.toFixed(1)}</div>
                    </div>
                    <div>
                      <div className="text-gray-600">Error Rate</div>
                      <div className="font-bold">{(apiHealth.error_rate * 100).toFixed(2)}%</div>
                    </div>
                    <div>
                      <div className="text-gray-600">Active Connections</div>
                      <div className="font-bold">{apiHealth.active_connections}</div>
                    </div>
                  </div>

                  <div className="pt-2 border-t">
                    <div className="text-sm text-gray-600">
                      Uptime: {formatUptime(apiHealth.uptime_seconds)}
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-center py-4 text-gray-500">
                  API health data unavailable
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Network I/O */}
      {systemMetrics?.network_io && (
        <Card>
          <CardHeader>
            <CardTitle>Network I/O</CardTitle>
            <CardDescription>
              Network traffic statistics
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <div className="text-sm text-gray-600 mb-1">Bytes Sent</div>
                <div className="text-2xl font-bold">{formatBytes(systemMetrics.network_io.bytes_sent)}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600 mb-1">Bytes Received</div>
                <div className="text-2xl font-bold">{formatBytes(systemMetrics.network_io.bytes_recv)}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Last Update */}
      {lastUpdate && (
        <div className="text-xs text-gray-500 text-center">
          Last updated: {lastUpdate.toLocaleString()}
        </div>
      )}
    </div>
  );
}