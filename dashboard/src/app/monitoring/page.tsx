/**
 * System Monitoring Dashboard Page
 * 
 * Provides comprehensive system monitoring including:
 * - Live API health status and resource monitoring
 * - CPU, memory, and GPU metrics visualization
 * - Error log display with filtering and search
 * - Alert system for system health issues
 * - System performance visualization with historical trends
 */

'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
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
  Search,
  Filter,
  TrendingUp,
  TrendingDown,
  Minus,
  Bell,
  BellOff,
  Eye,
  EyeOff
} from 'lucide-react';
import { useWebSocket, useSystemHealth } from '@/hooks/useWebSocket';
import { useApi } from '@/hooks/useApi';
import { apiClient } from '@/services/api';
import type { 
  SystemMetrics, 
  APIHealthMetrics, 
  ErrorLog, 
  SystemAlert, 
  GPUMetrics 
} from '@/types';

export default function MonitoringPage() {
  const { isConnected } = useWebSocket();
  const { systemHealth, alerts: wsAlerts, clearAlerts } = useSystemHealth();
  
  // State for monitoring data
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [apiHealth, setApiHealth] = useState<APIHealthMetrics | null>(null);
  const [gpuMetrics, setGPUMetrics] = useState<GPUMetrics | null>(null);
  const [errorLogs, setErrorLogs] = useState<ErrorLog[]>([]);
  const [systemAlerts, setSystemAlerts] = useState<SystemAlert[]>([]);
  const [performanceTrends, setPerformanceTrends] = useState<{
    timestamps: string[];
    cpu_usage: number[];
    memory_usage: number[];
    api_response_time: number[];
    error_rate: number[];
  } | null>(null);

  // Filter states
  const [logFilter, setLogFilter] = useState({
    level: 'all',
    search: '',
    limit: 100
  });
  const [alertFilter, setAlertFilter] = useState({
    resolved: false,
    severity: 'all',
    type: 'all'
  });

  // Auto-refresh state
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000); // 5 seconds

  // Fetch monitoring data
  const fetchMonitoringData = async () => {
    try {
      // Fetch system metrics
      const systemData = await apiClient.getSystemMetrics();
      setSystemMetrics(systemData);

      // Fetch API health
      const apiData = await apiClient.getAPIHealthMetrics();
      setApiHealth(apiData);

      // Fetch GPU metrics (optional)
      const gpuData = await apiClient.getGPUMetrics();
      setGPUMetrics(gpuData);

      // Fetch error logs
      const logs = await apiClient.getErrorLogs({
        limit: logFilter.limit,
        level: logFilter.level === 'all' ? undefined : logFilter.level as any,
        search: logFilter.search || undefined
      });
      setErrorLogs(logs);

      // Fetch system alerts
      const alerts = await apiClient.getSystemAlerts({
        resolved: alertFilter.resolved,
        severity: alertFilter.severity === 'all' ? undefined : alertFilter.severity as any,
        type: alertFilter.type === 'all' ? undefined : alertFilter.type as any
      });
      setSystemAlerts(alerts);

      // Fetch performance trends
      const trends = await apiClient.getPerformanceTrends({ hours: 24, interval: 'hour' });
      setPerformanceTrends(trends);

    } catch (error) {
      console.error('Failed to fetch monitoring data:', error);
    }
  };

  // Auto-refresh effect
  useEffect(() => {
    fetchMonitoringData();

    if (autoRefresh) {
      const interval = setInterval(fetchMonitoringData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval, logFilter, alertFilter]);

  // Helper functions
  const formatBytes = (bytes: number) => {
    const gb = bytes / (1024 * 1024 * 1024);
    return `${gb.toFixed(1)} GB`;
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
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

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'bg-blue-100 text-blue-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'high': return 'bg-orange-100 text-orange-800';
      case 'critical': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getTrendIcon = (current: number, previous: number) => {
    if (current > previous) return <TrendingUp className="h-4 w-4 text-red-500" />;
    if (current < previous) return <TrendingDown className="h-4 w-4 text-green-500" />;
    return <Minus className="h-4 w-4 text-gray-500" />;
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">System Monitoring</h1>
          <p className="text-gray-600">
            Real-time system health, performance metrics, and error monitoring
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={fetchMonitoringData}
            disabled={!isConnected}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button
            variant={autoRefresh ? "default" : "outline"}
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            {autoRefresh ? <Bell className="h-4 w-4 mr-2" /> : <BellOff className="h-4 w-4 mr-2" />}
            Auto Refresh
          </Button>
        </div>
      </div>

      {/* Connection Status */}
      {!isConnected && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Connection Warning</AlertTitle>
          <AlertDescription>
            Not connected to real-time monitoring. Some data may be outdated.
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="system">System</TabsTrigger>
          <TabsTrigger value="api">API Health</TabsTrigger>
          <TabsTrigger value="logs">Error Logs</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          {/* Status Cards */}
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
                <div className="flex items-center justify-between">
                  <span className={`text-lg font-bold ${getStatusColor(apiHealth?.status || 'unknown')}`}>
                    {apiHealth?.status?.toUpperCase() || 'UNKNOWN'}
                  </span>
                  {getStatusIcon(apiHealth?.status || 'unknown')}
                </div>
                {apiHealth && (
                  <div className="text-xs text-gray-500 mt-1">
                    {apiHealth.response_time_ms.toFixed(1)}ms avg response
                  </div>
                )}
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
                      {systemMetrics?.cpu_usage?.toFixed(1) || '0'}%
                    </span>
                    {systemMetrics && getTrendIcon(systemMetrics.cpu_usage, 50)}
                  </div>
                  <Progress 
                    value={systemMetrics?.cpu_usage || 0} 
                    className="h-2"
                  />
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
                      {systemMetrics ? ((systemMetrics.memory_usage / systemMetrics.memory_total) * 100).toFixed(1) : '0'}%
                    </span>
                    {systemMetrics && getTrendIcon(
                      (systemMetrics.memory_usage / systemMetrics.memory_total) * 100, 
                      50
                    )}
                  </div>
                  <Progress 
                    value={systemMetrics ? (systemMetrics.memory_usage / systemMetrics.memory_total) * 100 : 0} 
                    className="h-2"
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
                    <Progress value={gpuMetrics.utilization} className="h-2" />
                    <div className="text-xs text-gray-500">
                      {gpuMetrics.temperature}°C
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

          {/* Recent Alerts */}
          {systemAlerts.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  Recent Alerts
                </CardTitle>
                <CardDescription>
                  Latest system alerts and warnings
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {systemAlerts.slice(0, 5).map((alert) => (
                    <div key={alert.id} className="flex items-start gap-3 p-3 border rounded-lg">
                      <Badge className={getSeverityColor(alert.severity)}>
                        {alert.severity}
                      </Badge>
                      <div className="flex-1">
                        <div className="font-medium">{alert.title}</div>
                        <div className="text-sm text-gray-600">{alert.message}</div>
                        <div className="text-xs text-gray-500 mt-1">
                          {new Date(alert.timestamp).toLocaleString()}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* System Tab */}
        <TabsContent value="system" className="space-y-6">
          {systemMetrics && (
            <>
              {/* System Overview */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Cpu className="h-5 w-5" />
                      CPU Metrics
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Usage</span>
                        <span>{systemMetrics.cpu_usage.toFixed(1)}%</span>
                      </div>
                      <Progress value={systemMetrics.cpu_usage} className="h-2" />
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Load Average</div>
                      <div className="text-lg font-mono">
                        {systemMetrics.load_average.map(load => load.toFixed(2)).join(', ')}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <HardDrive className="h-5 w-5" />
                      Memory Metrics
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Usage</span>
                        <span>{((systemMetrics.memory_usage / systemMetrics.memory_total) * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={(systemMetrics.memory_usage / systemMetrics.memory_total) * 100} className="h-2" />
                    </div>
                    <div className="text-sm space-y-1">
                      <div className="flex justify-between">
                        <span>Used:</span>
                        <span>{formatBytes(systemMetrics.memory_usage)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Total:</span>
                        <span>{formatBytes(systemMetrics.memory_total)}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Server className="h-5 w-5" />
                      Disk Metrics
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Usage</span>
                        <span>{((systemMetrics.disk_usage / systemMetrics.disk_total) * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={(systemMetrics.disk_usage / systemMetrics.disk_total) * 100} className="h-2" />
                    </div>
                    <div className="text-sm space-y-1">
                      <div className="flex justify-between">
                        <span>Used:</span>
                        <span>{formatBytes(systemMetrics.disk_usage)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Total:</span>
                        <span>{formatBytes(systemMetrics.disk_total)}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Network I/O */}
              <Card>
                <CardHeader>
                  <CardTitle>Network I/O</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm text-gray-600">Bytes Sent</div>
                      <div className="text-lg font-mono">{formatBytes(systemMetrics.network_io.bytes_sent)}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Bytes Received</div>
                      <div className="text-lg font-mono">{formatBytes(systemMetrics.network_io.bytes_recv)}</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </TabsContent>

        {/* API Health Tab */}
        <TabsContent value="api" className="space-y-6">
          {apiHealth && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Activity className="h-5 w-5" />
                      API Status
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-2 mb-4">
                      {getStatusIcon(apiHealth.status)}
                      <span className={`text-lg font-bold ${getStatusColor(apiHealth.status)}`}>
                        {apiHealth.status.toUpperCase()}
                      </span>
                    </div>
                    <div className="text-sm text-gray-600">
                      Uptime: {formatUptime(apiHealth.uptime_seconds)}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Performance</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div>
                      <div className="text-sm text-gray-600">Response Time</div>
                      <div className="text-lg font-bold">{apiHealth.response_time_ms.toFixed(1)}ms</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Requests/sec</div>
                      <div className="text-lg font-bold">{apiHealth.requests_per_second.toFixed(1)}</div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Error Metrics</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div>
                      <div className="text-sm text-gray-600">Error Rate</div>
                      <div className="text-lg font-bold">{(apiHealth.error_rate * 100).toFixed(2)}%</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Active Connections</div>
                      <div className="text-lg font-bold">{apiHealth.active_connections}</div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </>
          )}
        </TabsContent>

        {/* Error Logs Tab */}
        <TabsContent value="logs" className="space-y-6">
          {/* Log Filters */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Filter className="h-5 w-5" />
                Log Filters
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex gap-4">
                <div className="flex-1">
                  <Input
                    placeholder="Search logs..."
                    value={logFilter.search}
                    onChange={(e) => setLogFilter(prev => ({ ...prev, search: e.target.value }))}
                    className="w-full"
                  />
                </div>
                <Select
                  value={logFilter.level}
                  onValueChange={(value) => setLogFilter(prev => ({ ...prev, level: value }))}
                >
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Levels</SelectItem>
                    <SelectItem value="error">Error</SelectItem>
                    <SelectItem value="warning">Warning</SelectItem>
                    <SelectItem value="info">Info</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Error Logs */}
          <Card>
            <CardHeader>
              <CardTitle>Error Logs</CardTitle>
              <CardDescription>
                Recent system errors and warnings
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {errorLogs.map((log) => (
                  <div key={log.id} className="p-3 border rounded-lg">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-2">
                        <Badge variant={log.level === 'error' ? 'destructive' : log.level === 'warning' ? 'secondary' : 'outline'}>
                          {log.level}
                        </Badge>
                        <span className="font-medium">{log.message}</span>
                      </div>
                      <span className="text-xs text-gray-500">
                        {new Date(log.timestamp).toLocaleString()}
                      </span>
                    </div>
                    {log.endpoint && (
                      <div className="text-sm text-gray-600 mt-1">
                        Endpoint: {log.endpoint}
                      </div>
                    )}
                    {log.error_type && (
                      <div className="text-sm text-gray-600">
                        Type: {log.error_type}
                      </div>
                    )}
                  </div>
                ))}
                {errorLogs.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    No error logs found
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Alerts Tab */}
        <TabsContent value="alerts" className="space-y-6">
          {/* Alert Filters */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Filter className="h-5 w-5" />
                Alert Filters
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex gap-4">
                <Select
                  value={alertFilter.severity}
                  onValueChange={(value) => setAlertFilter(prev => ({ ...prev, severity: value }))}
                >
                  <SelectTrigger className="w-40">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Severities</SelectItem>
                    <SelectItem value="critical">Critical</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="low">Low</SelectItem>
                  </SelectContent>
                </Select>
                <Select
                  value={alertFilter.type}
                  onValueChange={(value) => setAlertFilter(prev => ({ ...prev, type: value }))}
                >
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="system">System</SelectItem>
                    <SelectItem value="api">API</SelectItem>
                    <SelectItem value="model">Model</SelectItem>
                    <SelectItem value="gpu">GPU</SelectItem>
                  </SelectContent>
                </Select>
                <Button
                  variant="outline"
                  onClick={() => setAlertFilter(prev => ({ ...prev, resolved: !prev.resolved }))}
                >
                  {alertFilter.resolved ? <EyeOff className="h-4 w-4 mr-2" /> : <Eye className="h-4 w-4 mr-2" />}
                  {alertFilter.resolved ? 'Hide Resolved' : 'Show Resolved'}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* System Alerts */}
          <Card>
            <CardHeader>
              <CardTitle>System Alerts</CardTitle>
              <CardDescription>
                Active system alerts and notifications
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {systemAlerts.map((alert) => (
                  <div key={alert.id} className="p-4 border rounded-lg">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-3">
                        <Badge className={getSeverityColor(alert.severity)}>
                          {alert.severity}
                        </Badge>
                        <div>
                          <div className="font-medium">{alert.title}</div>
                          <div className="text-sm text-gray-600 mt-1">{alert.message}</div>
                          <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                            <span>Component: {alert.component}</span>
                            <span>Type: {alert.type}</span>
                            <span>{new Date(alert.timestamp).toLocaleString()}</span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {alert.resolved ? (
                          <Badge variant="outline" className="text-green-600">
                            Resolved
                          </Badge>
                        ) : (
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => apiClient.resolveAlert(alert.id)}
                          >
                            Resolve
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                {systemAlerts.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    No alerts found
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}