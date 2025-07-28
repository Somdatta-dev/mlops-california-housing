'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts';
import { 
  TrendingDown, 
  TrendingUp, 
  Activity, 
  Clock, 
  Target,
  Download,
  RefreshCw
} from 'lucide-react';
import { TrainingJob } from '@/types';

interface TrainingProgressChartProps {
  trainingJob: TrainingJob | null;
  isConnected: boolean;
}

export function TrainingProgressChart({ trainingJob, isConnected }: TrainingProgressChartProps) {
  const [chartType, setChartType] = useState<'loss' | 'accuracy' | 'learning_rate'>('loss');
  const [timeRange, setTimeRange] = useState<'all' | 'last_100' | 'last_50' | 'last_20'>('all');

  // Generate mock training data for demonstration
  const generateTrainingData = () => {
    if (!trainingJob?.metrics) return [];

    const { train_loss, val_loss, epochs, timestamps } = trainingJob.metrics;
    
    return epochs.map((epoch, index) => ({
      epoch,
      timestamp: timestamps[index],
      train_loss: train_loss[index],
      val_loss: val_loss[index],
      train_accuracy: trainingJob.metrics?.train_accuracy?.[index],
      val_accuracy: trainingJob.metrics?.val_accuracy?.[index],
      learning_rate: trainingJob.metrics?.learning_rate?.[index],
    }));
  };

  const filterDataByTimeRange = (data: any[]) => {
    switch (timeRange) {
      case 'last_20':
        return data.slice(-20);
      case 'last_50':
        return data.slice(-50);
      case 'last_100':
        return data.slice(-100);
      default:
        return data;
    }
  };

  const trainingData = filterDataByTimeRange(generateTrainingData());

  const getCurrentMetrics = () => {
    if (!trainingJob || !trainingData.length) return null;

    const latest = trainingData[trainingData.length - 1];
    const previous = trainingData.length > 1 ? trainingData[trainingData.length - 2] : null;

    return {
      current_loss: latest.train_loss,
      val_loss: latest.val_loss,
      loss_trend: previous ? latest.train_loss - previous.train_loss : 0,
      val_loss_trend: previous ? latest.val_loss - previous.val_loss : 0,
      current_accuracy: latest.train_accuracy,
      val_accuracy: latest.val_accuracy,
      learning_rate: latest.learning_rate,
    };
  };

  const metrics = getCurrentMetrics();

  const exportData = () => {
    if (!trainingData.length) return;

    const csvContent = [
      ['Epoch', 'Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy', 'Learning Rate', 'Timestamp'].join(','),
      ...trainingData.map(row => [
        row.epoch,
        row.train_loss?.toFixed(6) || '',
        row.val_loss?.toFixed(6) || '',
        row.train_accuracy?.toFixed(4) || '',
        row.val_accuracy?.toFixed(4) || '',
        row.learning_rate?.toExponential(2) || '',
        row.timestamp
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `training_progress_${trainingJob?.id || 'data'}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!isConnected) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Training Progress
          </CardTitle>
          <CardDescription>Real-time training metrics and loss curves</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="text-center">
              <Activity className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">Not connected to training server</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!trainingJob) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Training Progress
          </CardTitle>
          <CardDescription>Real-time training metrics and loss curves</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="text-center">
              <Clock className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No active training job</p>
              <p className="text-sm text-gray-400 mt-2">Start a training job to see progress metrics</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Metrics Overview */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Training Loss</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold">{metrics.current_loss?.toFixed(6)}</span>
                <div className="flex items-center gap-1">
                  {metrics.loss_trend < 0 ? (
                    <TrendingDown className="h-4 w-4 text-green-600" />
                  ) : (
                    <TrendingUp className="h-4 w-4 text-red-600" />
                  )}
                  <span className={`text-sm ${metrics.loss_trend < 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {Math.abs(metrics.loss_trend).toFixed(6)}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Validation Loss</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold">{metrics.val_loss?.toFixed(6)}</span>
                <div className="flex items-center gap-1">
                  {metrics.val_loss_trend < 0 ? (
                    <TrendingDown className="h-4 w-4 text-green-600" />
                  ) : (
                    <TrendingUp className="h-4 w-4 text-red-600" />
                  )}
                  <span className={`text-sm ${metrics.val_loss_trend < 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {Math.abs(metrics.val_loss_trend).toFixed(6)}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          {metrics.current_accuracy && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Training Accuracy</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <span className="text-2xl font-bold">{(metrics.current_accuracy * 100).toFixed(2)}%</span>
                  <Target className="h-4 w-4 text-blue-600" />
                </div>
              </CardContent>
            </Card>
          )}

          {metrics.learning_rate && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Learning Rate</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <span className="text-2xl font-bold">{metrics.learning_rate.toExponential(2)}</span>
                  <Activity className="h-4 w-4 text-purple-600" />
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Training Progress Chart */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Training Progress
              </CardTitle>
              <CardDescription>
                {trainingJob.model_type.replace('_', ' ').toUpperCase()} - 
                Epoch {trainingJob.current_epoch || 0} of {trainingJob.total_epochs || 0}
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Select value={timeRange} onValueChange={(value: typeof timeRange) => setTimeRange(value)}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Data</SelectItem>
                  <SelectItem value="last_100">Last 100</SelectItem>
                  <SelectItem value="last_50">Last 50</SelectItem>
                  <SelectItem value="last_20">Last 20</SelectItem>
                </SelectContent>
              </Select>
              <Select value={chartType} onValueChange={(value: typeof chartType) => setChartType(value)}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="loss">Loss</SelectItem>
                  <SelectItem value="accuracy">Accuracy</SelectItem>
                  <SelectItem value="learning_rate">Learning Rate</SelectItem>
                </SelectContent>
              </Select>
              <Button variant="outline" size="sm" onClick={exportData}>
                <Download className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {trainingData.length > 0 ? (
            <div className="h-96">
              {chartType === 'loss' && (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trainingData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <Tooltip 
                      labelFormatter={(value) => `Epoch ${value}`}
                      formatter={(value: number, name: string) => [
                        value?.toFixed(6), 
                        name === 'train_loss' ? 'Training Loss' : 'Validation Loss'
                      ]}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="train_loss" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      name="Training Loss"
                      dot={false}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="val_loss" 
                      stroke="#ef4444" 
                      strokeWidth={2}
                      name="Validation Loss"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
              {chartType === 'accuracy' && (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trainingData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <Tooltip 
                      labelFormatter={(value) => `Epoch ${value}`}
                      formatter={(value: number, name: string) => [
                        `${(value * 100).toFixed(2)}%`, 
                        name === 'train_accuracy' ? 'Training Accuracy' : 'Validation Accuracy'
                      ]}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="train_accuracy" 
                      stroke="#10b981" 
                      strokeWidth={2}
                      name="Training Accuracy"
                      dot={false}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="val_accuracy" 
                      stroke="#f59e0b" 
                      strokeWidth={2}
                      name="Validation Accuracy"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
              {chartType === 'learning_rate' && (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={trainingData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <Tooltip 
                      labelFormatter={(value) => `Epoch ${value}`}
                      formatter={(value: number) => [value?.toExponential(2), 'Learning Rate']}
                    />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="learning_rate" 
                      stroke="#8b5cf6" 
                      fill="#8b5cf6"
                      fillOpacity={0.3}
                      name="Learning Rate"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
          ) : (
            <div className="flex items-center justify-center h-96">
              <div className="text-center">
                <RefreshCw className="h-12 w-12 text-gray-400 mx-auto mb-4 animate-spin" />
                <p className="text-gray-500">Waiting for training data...</p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Training Status */}
      <Card>
        <CardHeader>
          <CardTitle>Training Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-sm text-gray-500">Status</div>
              <Badge variant={trainingJob.status === 'training' ? 'default' : 'secondary'}>
                {trainingJob.status.toUpperCase()}
              </Badge>
            </div>
            <div>
              <div className="text-sm text-gray-500">Model Type</div>
              <div className="font-medium">{trainingJob.model_type.replace('_', ' ').toUpperCase()}</div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Progress</div>
              <div className="font-medium">{Math.round(trainingJob.progress)}%</div>
            </div>
            <div>
              <div className="text-sm text-gray-500">GPU Enabled</div>
              <div className="font-medium">{trainingJob.config.gpu_enabled ? 'Yes' : 'No'}</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}