/**
 * Dashboard overview component showing key metrics and status
 */

'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Activity, 
  Brain, 
  Zap, 
  Database,
  TrendingUp,
  Clock,
  Cpu,
  HardDrive
} from 'lucide-react';
import { useHealthStatus, useModelInfo } from '@/hooks/useApi';
import { useGPUMetrics, useSystemHealth } from '@/hooks/useWebSocket';
import { formatNumber, formatBytes, formatGPUUtilization, formatTemperature } from '@/utils/format';

export function DashboardOverview() {
  const { data: healthStatus, loading: healthLoading } = useHealthStatus();
  const { data: modelInfo } = useModelInfo();
  const { gpuMetrics } = useGPUMetrics();
  const { systemHealth } = useSystemHealth();

  const MetricCard = ({ 
    title, 
    value, 
    description, 
    icon: Icon, 
    trend, 
    loading = false 
  }: {
    title: string;
    value: string | number;
    description: string;
    icon: React.ComponentType<{ className?: string }>;
    trend?: 'up' | 'down' | 'stable';
    loading?: boolean;
  }) => (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">
          {loading ? '...' : value}
        </div>
        <p className="text-xs text-muted-foreground">
          {description}
        </p>
        {trend && (
          <div className="flex items-center pt-1">
            <TrendingUp className={`h-3 w-3 ${
              trend === 'up' ? 'text-green-600' : 
              trend === 'down' ? 'text-red-600' : 
              'text-gray-600'
            }`} />
            <span className="text-xs text-muted-foreground ml-1">
              {trend === 'up' ? 'Increasing' : 
               trend === 'down' ? 'Decreasing' : 
               'Stable'}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
        <p className="text-muted-foreground">
          Overview of your MLOps platform performance and status
        </p>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="System Status"
          value={healthStatus?.status || 'Unknown'}
          description="Overall system health"
          icon={Activity}
          loading={healthLoading}
        />
        
        <MetricCard
          title="Model Status"
          value={healthStatus?.model_loaded ? 'Loaded' : 'Not Loaded'}
          description="Current model availability"
          icon={Brain}
          loading={healthLoading}
        />
        
        <MetricCard
          title="GPU Status"
          value={healthStatus?.gpu_available ? 'Available' : 'Unavailable'}
          description="GPU acceleration status"
          icon={Zap}
          loading={healthLoading}
        />
        
        <MetricCard
          title="Uptime"
          value={healthStatus ? `${Math.floor(healthStatus.uptime / 3600)}h` : '0h'}
          description="System uptime"
          icon={Clock}
          loading={healthLoading}
        />
      </div>

      {/* Model Information */}
      {modelInfo && (
        <Card>
          <CardHeader>
            <CardTitle>Current Model</CardTitle>
            <CardDescription>
              Information about the currently loaded model
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <div>
                <div className="text-sm font-medium text-muted-foreground">Model Type</div>
                <div className="text-lg font-semibold">{modelInfo.model_type}</div>
              </div>
              <div>
                <div className="text-sm font-medium text-muted-foreground">Version</div>
                <div className="text-lg font-semibold">{modelInfo.model_version}</div>
              </div>
              <div>
                <div className="text-sm font-medium text-muted-foreground">RMSE</div>
                <div className="text-lg font-semibold">
                  {formatNumber(modelInfo.performance_metrics.rmse, 4)}
                </div>
              </div>
              <div>
                <div className="text-sm font-medium text-muted-foreground">R² Score</div>
                <div className="text-lg font-semibold">
                  {formatNumber(modelInfo.performance_metrics.r2, 3)}
                </div>
              </div>
            </div>
            
            <div className="mt-4 flex items-center space-x-2">
              <Badge variant={modelInfo.gpu_accelerated ? 'default' : 'secondary'}>
                {modelInfo.gpu_accelerated ? 'GPU Accelerated' : 'CPU Only'}
              </Badge>
              <span className="text-sm text-muted-foreground">
                Trained: {new Date(modelInfo.training_date).toLocaleDateString()}
              </span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* GPU Metrics */}
      {gpuMetrics && (
        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>GPU Utilization</CardTitle>
              <CardDescription>Current GPU usage and performance</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex items-center justify-between text-sm">
                  <span>Utilization</span>
                  <span>{formatGPUUtilization(gpuMetrics.utilization)}</span>
                </div>
                <Progress value={gpuMetrics.utilization} className="mt-2" />
              </div>
              
              <div>
                <div className="flex items-center justify-between text-sm">
                  <span>Memory Usage</span>
                  <span>
                    {formatBytes(gpuMetrics.memory_used)} / {formatBytes(gpuMetrics.memory_total)}
                  </span>
                </div>
                <Progress 
                  value={(gpuMetrics.memory_used / gpuMetrics.memory_total) * 100} 
                  className="mt-2" 
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>GPU Status</CardTitle>
              <CardDescription>Temperature and power consumption</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm font-medium text-muted-foreground">Temperature</div>
                  <div className="text-2xl font-bold">
                    {formatTemperature(gpuMetrics.temperature)}
                  </div>
                </div>
                <div>
                  <div className="text-sm font-medium text-muted-foreground">Power Usage</div>
                  <div className="text-2xl font-bold">
                    {formatNumber(gpuMetrics.power_usage / 1000, 1)}W
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* System Health */}
      {systemHealth && (
        <Card>
          <CardHeader>
            <CardTitle>System Resources</CardTitle>
            <CardDescription>Current system resource utilization</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              <div>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="flex items-center">
                    <Cpu className="h-4 w-4 mr-1" />
                    CPU Usage
                  </span>
                  <span>{formatNumber(systemHealth.cpu_usage, 1)}%</span>
                </div>
                <Progress value={systemHealth.cpu_usage} />
              </div>
              
              <div>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="flex items-center">
                    <HardDrive className="h-4 w-4 mr-1" />
                    Memory Usage
                  </span>
                  <span>{formatNumber(systemHealth.memory_usage, 1)}%</span>
                </div>
                <Progress value={systemHealth.memory_usage} />
              </div>
              
              <div>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="flex items-center">
                    <Database className="h-4 w-4 mr-1" />
                    Disk Usage
                  </span>
                  <span>{formatNumber(systemHealth.disk_usage, 1)}%</span>
                </div>
                <Progress value={systemHealth.disk_usage} />
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}