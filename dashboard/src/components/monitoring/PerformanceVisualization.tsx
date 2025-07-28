/**
 * PerformanceVisualization Component
 * 
 * Displays system performance visualization with historical trends and charts
 */

'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar
} from 'recharts';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Cpu, 
  HardDrive, 
  Zap, 
  Clock,
  RefreshCw,
  Download
} from 'lucide-react';
import { useWebSocket, useSystemHealth, useGPUMetrics } from '@/hooks/useWebSocket';
import { apiClient } from '@/services/api';

interface PerformanceVisualizationProps {
  className?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

interface PerformanceData {
  timestamp: string;
  cpu_usage: number;
  memory_usage: number;
  api_response_time: number;
  error_rate: number;
  gpu_utilization?: number;
  gpu_temperature?: number;
}

export function PerformanceVisualization({ 
  className = '', 
  autoRefresh = true,
  refreshInterval = 60000 // 1 minute
}: PerformanceVisualizationProps) {
  const { isConnected } = useWebSocket();
  const { systemHealth } = useSystemHealth();
  const { gpuMetrics, metricsHistory } = useGPUMetrics();
  
  // State
  const [performanceData, setPerformanceData] = useState<PerformanceData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [timeRange, setTimeRange] = useState('24h');
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['cpu_usage', 'memory_usage', 'api_response_time']);

  // Fetch performance trends
  const fetchPerformanceData = async () => {
    setIsLoading(true);
    try {
      const hours = timeRange === '1h' ? 1 : timeRange === '6h' ? 6 : timeRange === '24h' ? 24 : 168; // 1 week
      const interval = hours <= 6 ? 'minute' : 'hour';
      
      const trends = await apiClient.getPerformanceTrends({ hours, interval });
      
      // Transform data for charts
      const transformedData: PerformanceData[] = trends.timestamps.map((timestamp, index) => ({
        timestamp,
        cpu_usage: trends.cpu_usage[index] || 0,
        memory_usage: trends.memory_usage[index] || 0,
        api_response_time: trends.api_response_time[index] || 0,
        error_rate: trends.error_rate[index] || 0,
        gpu_utilization: metricsHistory[index]?.utilization,
        gpu_temperature: metricsHistory[index]?.temperature
      }));
      
      setPerformanceData(transformedData);
    } catch (error) {
      console.error('Failed to fetch performance data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-refresh effect
  useEffect(() => {
    fetchPerformanceData();

    if (autoRefresh) {
      const interval = setInterval(fetchPerformanceData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval, timeRange]);

  // Real-time data integration
  useEffect(() => {
    if (systemHealth && performanceData.length > 0) {
      const now = new Date().toISOString();
      const newDataPoint: PerformanceData = {
        timestamp: now,
        cpu_usage: systemHealth.cpu_usage,
        memory_usage: systemHealth.memory_usage,
        api_response_time: systemHealth.api_response_time,
        error_rate: 0, // Would need to calculate from recent errors
        gpu_utilization: gpuMetrics?.utilization,
        gpu_temperature: gpuMetrics?.temperature
      };

      setPerformanceData(prev => {
        const updated = [...prev.slice(-59), newDataPoint]; // Keep last 60 points
        return updated;
      });
    }
  }, [systemHealth, gpuMetrics]);

  // Helper functions
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    if (timeRange === '1h' || timeRange === '6h') {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    return date.toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit' });
  };

  const calculateTrend = (data: number[]) => {
    if (data.length < 2) return 0;
    const recent = data.slice(-5).reduce((sum, val) => sum + val, 0) / 5;
    const previous = data.slice(-10, -5).reduce((sum, val) => sum + val, 0) / 5;
    return ((recent - previous) / previous) * 100;
  };

  const getTrendIcon = (trend: number) => {
    if (trend > 5) return <TrendingUp className="h-4 w-4 text-red-500" />;
    if (trend < -5) return <TrendingDown className="h-4 w-4 text-green-500" />;
    return <Activity className="h-4 w-4 text-gray-500" />;
  };

  const exportData = () => {
    const csvContent = [
      ['Timestamp', 'CPU Usage (%)', 'Memory Usage (%)', 'API Response Time (ms)', 'Error Rate (%)', 'GPU Utilization (%)', 'GPU Temperature (°C)'],
      ...performanceData.map(d => [
        d.timestamp,
        d.cpu_usage.toFixed(2),
        d.memory_usage.toFixed(2),
        d.api_response_time.toFixed(2),
        (d.error_rate * 100).toFixed(2),
        d.gpu_utilization?.toFixed(2) || '',
        d.gpu_temperature?.toFixed(2) || ''
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `performance_data_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Calculate current trends
  const cpuTrend = calculateTrend(performanceData.map(d => d.cpu_usage));
  const memoryTrend = calculateTrend(performanceData.map(d => d.memory_usage));
  const apiTrend = calculateTrend(performanceData.map(d => d.api_response_time));

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Controls */}
      <Card>
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Performance Trends
            </span>
            <div className="flex items-center gap-2">
              {!isConnected && (
                <Badge variant="outline" className="text-yellow-600">
                  Offline
                </Badge>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={fetchPerformanceData}
                disabled={isLoading}
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={exportData}
                disabled={performanceData.length === 0}
              >
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1h">Last Hour</SelectItem>
                <SelectItem value="6h">Last 6 Hours</SelectItem>
                <SelectItem value="24h">Last 24 Hours</SelectItem>
                <SelectItem value="7d">Last 7 Days</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Trend Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-blue-500" />
                <span className="text-sm font-medium">CPU Trend</span>
              </div>
              <div className="flex items-center gap-2">
                {getTrendIcon(cpuTrend)}
                <span className={`text-sm font-bold ${
                  cpuTrend > 5 ? 'text-red-600' : cpuTrend < -5 ? 'text-green-600' : 'text-gray-600'
                }`}>
                  {cpuTrend > 0 ? '+' : ''}{cpuTrend.toFixed(1)}%
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <HardDrive className="h-4 w-4 text-green-500" />
                <span className="text-sm font-medium">Memory Trend</span>
              </div>
              <div className="flex items-center gap-2">
                {getTrendIcon(memoryTrend)}
                <span className={`text-sm font-bold ${
                  memoryTrend > 5 ? 'text-red-600' : memoryTrend < -5 ? 'text-green-600' : 'text-gray-600'
                }`}>
                  {memoryTrend > 0 ? '+' : ''}{memoryTrend.toFixed(1)}%
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Clock className="h-4 w-4 text-purple-500" />
                <span className="text-sm font-medium">API Response Trend</span>
              </div>
              <div className="flex items-center gap-2">
                {getTrendIcon(apiTrend)}
                <span className={`text-sm font-bold ${
                  apiTrend > 5 ? 'text-red-600' : apiTrend < -5 ? 'text-green-600' : 'text-gray-600'
                }`}>
                  {apiTrend > 0 ? '+' : ''}{apiTrend.toFixed(1)}%
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* System Performance Chart */}
      <Card>
        <CardHeader>
          <CardTitle>System Performance</CardTitle>
          <CardDescription>
            CPU and memory usage over time
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={formatTimestamp}
                  interval="preserveStartEnd"
                />
                <YAxis domain={[0, 100]} />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                  formatter={(value: number, name: string) => [
                    `${value.toFixed(1)}${name.includes('usage') ? '%' : name.includes('time') ? 'ms' : ''}`,
                    name.replace('_', ' ').toUpperCase()
                  ]}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="cpu_usage" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  name="CPU Usage"
                  dot={false}
                />
                <Line 
                  type="monotone" 
                  dataKey="memory_usage" 
                  stroke="#10b981" 
                  strokeWidth={2}
                  name="Memory Usage"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* API Performance Chart */}
      <Card>
        <CardHeader>
          <CardTitle>API Performance</CardTitle>
          <CardDescription>
            Response time and error rate trends
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={formatTimestamp}
                  interval="preserveStartEnd"
                />
                <YAxis yAxisId="left" orientation="left" />
                <YAxis yAxisId="right" orientation="right" domain={[0, 100]} />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                  formatter={(value: number, name: string) => [
                    `${value.toFixed(1)}${name.includes('time') ? 'ms' : '%'}`,
                    name.replace('_', ' ').toUpperCase()
                  ]}
                />
                <Legend />
                <Line 
                  yAxisId="left"
                  type="monotone" 
                  dataKey="api_response_time" 
                  stroke="#8b5cf6" 
                  strokeWidth={2}
                  name="Response Time (ms)"
                  dot={false}
                />
                <Line 
                  yAxisId="right"
                  type="monotone" 
                  dataKey="error_rate" 
                  stroke="#ef4444" 
                  strokeWidth={2}
                  name="Error Rate (%)"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* GPU Performance Chart */}
      {gpuMetrics && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5" />
              GPU Performance
            </CardTitle>
            <CardDescription>
              GPU utilization and temperature trends
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={performanceData.filter(d => d.gpu_utilization !== undefined)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={formatTimestamp}
                    interval="preserveStartEnd"
                  />
                  <YAxis yAxisId="left" domain={[0, 100]} />
                  <YAxis yAxisId="right" orientation="right" domain={[0, 100]} />
                  <Tooltip 
                    labelFormatter={(value) => new Date(value).toLocaleString()}
                    formatter={(value: number, name: string) => [
                      `${value.toFixed(1)}${name.includes('utilization') ? '%' : '°C'}`,
                      name.replace('_', ' ').toUpperCase()
                    ]}
                  />
                  <Legend />
                  <Line 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="gpu_utilization" 
                    stroke="#f59e0b" 
                    strokeWidth={2}
                    name="GPU Utilization (%)"
                    dot={false}
                  />
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="gpu_temperature" 
                    stroke="#dc2626" 
                    strokeWidth={2}
                    name="GPU Temperature (°C)"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      {/* No Data State */}
      {performanceData.length === 0 && !isLoading && (
        <Card>
          <CardContent className="p-8 text-center">
            <Activity className="h-12 w-12 mx-auto mb-4 text-gray-300" />
            <p className="text-gray-500 mb-2">No performance data available</p>
            <p className="text-sm text-gray-400">
              Performance data will appear here once the system starts collecting metrics
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}