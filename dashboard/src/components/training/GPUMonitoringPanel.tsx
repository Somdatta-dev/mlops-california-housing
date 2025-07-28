'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { 
  Zap, 
  Thermometer, 
  HardDrive, 
  Activity, 
  AlertTriangle,
  CheckCircle,
  XCircle 
} from 'lucide-react';
import { GPUMetrics } from '@/types';

interface GPUMonitoringPanelProps {
  gpuMetrics: GPUMetrics | null;
  isConnected: boolean;
}

export function GPUMonitoringPanel({ gpuMetrics, isConnected }: GPUMonitoringPanelProps) {
  const [historicalData, setHistoricalData] = useState<{
    utilization: number[];
    memory: number[];
    temperature: number[];
    power: number[];
    timestamps: string[];
  }>({
    utilization: [],
    memory: [],
    temperature: [],
    power: [],
    timestamps: [],
  });

  // Update historical data when new metrics arrive
  useEffect(() => {
    if (gpuMetrics) {
      const now = new Date().toLocaleTimeString();
      const maxDataPoints = 50;

      setHistoricalData(prev => ({
        utilization: [...prev.utilization.slice(-maxDataPoints + 1), gpuMetrics.utilization],
        memory: [...prev.memory.slice(-maxDataPoints + 1), (gpuMetrics.memory_used / gpuMetrics.memory_total) * 100],
        temperature: [...prev.temperature.slice(-maxDataPoints + 1), gpuMetrics.temperature],
        power: [...prev.power.slice(-maxDataPoints + 1), gpuMetrics.power_usage / 1000], // Convert to watts
        timestamps: [...prev.timestamps.slice(-maxDataPoints + 1), now],
      }));
    }
  }, [gpuMetrics]);

  const formatBytes = (bytes: number) => {
    const gb = bytes / (1024 * 1024 * 1024);
    return `${gb.toFixed(1)} GB`;
  };

  const getStatusColor = (value: number, thresholds: { warning: number; critical: number }) => {
    if (value >= thresholds.critical) return 'text-red-600';
    if (value >= thresholds.warning) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getStatusIcon = (value: number, thresholds: { warning: number; critical: number }) => {
    if (value >= thresholds.critical) return <XCircle className="h-4 w-4 text-red-600" />;
    if (value >= thresholds.warning) return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
    return <CheckCircle className="h-4 w-4 text-green-600" />;
  };

  const getProgressColor = (value: number, thresholds: { warning: number; critical: number }) => {
    if (value >= thresholds.critical) return 'bg-red-500';
    if (value >= thresholds.warning) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  if (!isConnected) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            GPU Monitoring
          </CardTitle>
          <CardDescription>Real-time GPU utilization and performance metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="text-center">
              <XCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">Not connected to monitoring server</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!gpuMetrics) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            GPU Monitoring
          </CardTitle>
          <CardDescription>Real-time GPU utilization and performance metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-500">Loading GPU metrics...</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const memoryUsagePercent = (gpuMetrics.memory_used / gpuMetrics.memory_total) * 100;

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* GPU Utilization */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Zap className="h-4 w-4" />
              GPU Utilization
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold">{gpuMetrics.utilization}%</span>
                {getStatusIcon(gpuMetrics.utilization, { warning: 80, critical: 95 })}
              </div>
              <Progress 
                value={gpuMetrics.utilization} 
                className="h-2"
                style={{
                  '--progress-background': getProgressColor(gpuMetrics.utilization, { warning: 80, critical: 95 })
                } as React.CSSProperties}
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
                <span className="text-2xl font-bold">{Math.round(memoryUsagePercent)}%</span>
                {getStatusIcon(memoryUsagePercent, { warning: 80, critical: 95 })}
              </div>
              <Progress 
                value={memoryUsagePercent} 
                className="h-2"
                style={{
                  '--progress-background': getProgressColor(memoryUsagePercent, { warning: 80, critical: 95 })
                } as React.CSSProperties}
              />
              <div className="text-xs text-gray-500">
                {formatBytes(gpuMetrics.memory_used)} / {formatBytes(gpuMetrics.memory_total)}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Temperature */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Thermometer className="h-4 w-4" />
              Temperature
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold">{gpuMetrics.temperature}°C</span>
                {getStatusIcon(gpuMetrics.temperature, { warning: 75, critical: 85 })}
              </div>
              <Progress 
                value={(gpuMetrics.temperature / 100) * 100} 
                className="h-2"
                style={{
                  '--progress-background': getProgressColor(gpuMetrics.temperature, { warning: 75, critical: 85 })
                } as React.CSSProperties}
              />
            </div>
          </CardContent>
        </Card>

        {/* Power Usage */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Power Usage
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold">{Math.round(gpuMetrics.power_usage / 1000)}W</span>
                {getStatusIcon(gpuMetrics.power_usage / 1000, { warning: 250, critical: 300 })}
              </div>
              <Progress 
                value={(gpuMetrics.power_usage / 1000 / 350) * 100} 
                className="h-2"
                style={{
                  '--progress-background': getProgressColor(gpuMetrics.power_usage / 1000, { warning: 250, critical: 300 })
                } as React.CSSProperties}
              />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Monitoring */}
      <Card>
        <CardHeader>
          <CardTitle>Detailed GPU Metrics</CardTitle>
          <CardDescription>
            Real-time monitoring with historical data and performance insights
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Status Indicators */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex items-center gap-2">
              <Badge variant={gpuMetrics.utilization > 80 ? 'destructive' : 'secondary'}>
                Utilization
              </Badge>
              <span className={getStatusColor(gpuMetrics.utilization, { warning: 80, critical: 95 })}>
                {gpuMetrics.utilization > 80 ? 'High' : gpuMetrics.utilization > 50 ? 'Medium' : 'Low'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={memoryUsagePercent > 80 ? 'destructive' : 'secondary'}>
                Memory
              </Badge>
              <span className={getStatusColor(memoryUsagePercent, { warning: 80, critical: 95 })}>
                {memoryUsagePercent > 80 ? 'High' : memoryUsagePercent > 50 ? 'Medium' : 'Low'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={gpuMetrics.temperature > 75 ? 'destructive' : 'secondary'}>
                Temperature
              </Badge>
              <span className={getStatusColor(gpuMetrics.temperature, { warning: 75, critical: 85 })}>
                {gpuMetrics.temperature > 75 ? 'Hot' : gpuMetrics.temperature > 60 ? 'Warm' : 'Cool'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={gpuMetrics.power_usage / 1000 > 250 ? 'destructive' : 'secondary'}>
                Power
              </Badge>
              <span className={getStatusColor(gpuMetrics.power_usage / 1000, { warning: 250, critical: 300 })}>
                {gpuMetrics.power_usage / 1000 > 250 ? 'High' : gpuMetrics.power_usage / 1000 > 150 ? 'Medium' : 'Low'}
              </span>
            </div>
          </div>

          <Separator />

          {/* Performance Recommendations */}
          <div className="space-y-2">
            <h4 className="font-medium">Performance Recommendations</h4>
            <div className="text-sm text-gray-600 space-y-1">
              {gpuMetrics.utilization < 50 && (
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-yellow-500" />
                  GPU utilization is low. Consider increasing batch size or model complexity.
                </div>
              )}
              {memoryUsagePercent > 90 && (
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-red-500" />
                  GPU memory usage is very high. Consider reducing batch size or enabling gradient checkpointing.
                </div>
              )}
              {gpuMetrics.temperature > 80 && (
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-red-500" />
                  GPU temperature is high. Check cooling and consider reducing workload.
                </div>
              )}
              {gpuMetrics.utilization > 80 && memoryUsagePercent < 70 && (
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  Optimal GPU utilization with good memory efficiency.
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}