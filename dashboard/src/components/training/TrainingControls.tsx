'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Play, Pause, Square, RotateCcw, Zap, Clock, TrendingUp } from 'lucide-react';
import { TrainingJob, TrainingConfig } from '@/types';

interface TrainingControlsProps {
  activeJob: TrainingJob | null;
  onStartTraining: (config: TrainingConfig) => void;
  onPauseTraining: () => void;
  onStopTraining: () => void;
  onResumeTraining: () => void;
}

export function TrainingControls({
  activeJob,
  onStartTraining,
  onPauseTraining,
  onStopTraining,
  onResumeTraining,
}: TrainingControlsProps) {
  const [selectedModelType, setSelectedModelType] = useState<string>('xgboost');

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'training':
        return 'bg-blue-500';
      case 'paused':
        return 'bg-yellow-500';
      case 'completed':
        return 'bg-green-500';
      case 'error':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'training':
        return <Play className="h-4 w-4" />;
      case 'paused':
        return <Pause className="h-4 w-4" />;
      case 'completed':
        return <TrendingUp className="h-4 w-4" />;
      case 'error':
        return <Square className="h-4 w-4" />;
      default:
        return <Clock className="h-4 w-4" />;
    }
  };

  const handleQuickStart = () => {
    const quickConfig: TrainingConfig = {
      model_type: selectedModelType as any,
      hyperparameters: getDefaultHyperparameters(selectedModelType),
      gpu_enabled: true,
      mixed_precision: true,
      early_stopping: true,
      validation_split: 0.2,
    };
    onStartTraining(quickConfig);
  };

  const getDefaultHyperparameters = (modelType: string) => {
    switch (modelType) {
      case 'xgboost':
        return {
          max_depth: 6,
          n_estimators: 1000,
          learning_rate: 0.1,
          tree_method: 'gpu_hist',
        };
      case 'neural_network':
        return {
          hidden_layers: [512, 256, 128],
          batch_size: 1024,
          epochs: 100,
          learning_rate: 0.001,
        };
      case 'lightgbm':
        return {
          num_leaves: 31,
          n_estimators: 1000,
          learning_rate: 0.1,
          device: 'gpu',
        };
      case 'random_forest':
        return {
          n_estimators: 1000,
          max_depth: 10,
          min_samples_split: 2,
        };
      case 'linear_regression':
        return {
          fit_intercept: true,
          normalize: false,
        };
      default:
        return {};
    }
  };

  const formatDuration = (startTime?: string, endTime?: string) => {
    if (!startTime) return 'N/A';
    
    const start = new Date(startTime);
    const end = endTime ? new Date(endTime) : new Date();
    const duration = Math.floor((end.getTime() - start.getTime()) / 1000);
    
    const hours = Math.floor(duration / 3600);
    const minutes = Math.floor((duration % 3600) / 60);
    const seconds = duration % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${seconds}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds}s`;
    } else {
      return `${seconds}s`;
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="h-5 w-5" />
          Training Controls
        </CardTitle>
        <CardDescription>
          Start, pause, or stop model training jobs with GPU acceleration
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Current Job Status */}
        {activeJob && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="flex items-center gap-1">
                  {getStatusIcon(activeJob.status)}
                  {activeJob.status.toUpperCase()}
                </Badge>
                <span className="text-sm text-gray-600">
                  {activeJob.model_type.replace('_', ' ').toUpperCase()}
                </span>
              </div>
              <div className="text-sm text-gray-500">
                Duration: {formatDuration(activeJob.start_time, activeJob.end_time)}
              </div>
            </div>

            {/* Progress Bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progress</span>
                <span>{Math.round(activeJob.progress)}%</span>
              </div>
              <Progress value={activeJob.progress} className="h-2" />
              {activeJob.current_epoch && activeJob.total_epochs && (
                <div className="text-xs text-gray-500">
                  Epoch {activeJob.current_epoch} of {activeJob.total_epochs}
                </div>
              )}
            </div>

            {/* Current Metrics */}
            {activeJob.current_loss && (
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Current Loss:</span>
                  <div className="font-mono">{activeJob.current_loss.toFixed(6)}</div>
                </div>
                {activeJob.best_loss && (
                  <div>
                    <span className="text-gray-500">Best Loss:</span>
                    <div className="font-mono">{activeJob.best_loss.toFixed(6)}</div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Control Buttons */}
        <div className="space-y-4">
          {!activeJob || activeJob.status === 'idle' || activeJob.status === 'completed' || activeJob.status === 'error' ? (
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Quick Start Model</label>
                <Select value={selectedModelType} onValueChange={setSelectedModelType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select model type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="xgboost">XGBoost (GPU)</SelectItem>
                    <SelectItem value="neural_network">Neural Network (PyTorch)</SelectItem>
                    <SelectItem value="lightgbm">LightGBM (GPU)</SelectItem>
                    <SelectItem value="random_forest">Random Forest (cuML)</SelectItem>
                    <SelectItem value="linear_regression">Linear Regression (cuML)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button onClick={handleQuickStart} className="w-full" size="lg">
                <Play className="h-4 w-4 mr-2" />
                Start Training
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              {activeJob.status === 'training' && (
                <Button onClick={onPauseTraining} variant="outline">
                  <Pause className="h-4 w-4 mr-2" />
                  Pause
                </Button>
              )}
              {activeJob.status === 'paused' && (
                <Button onClick={onResumeTraining} variant="outline">
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Resume
                </Button>
              )}
              <Button onClick={onStopTraining} variant="destructive">
                <Square className="h-4 w-4 mr-2" />
                Stop
              </Button>
            </div>
          )}
        </div>

        {/* Training Info */}
        <div className="text-xs text-gray-500 space-y-1">
          <div>• GPU acceleration enabled by default</div>
          <div>• Mixed precision training for memory efficiency</div>
          <div>• Early stopping to prevent overfitting</div>
          <div>• Real-time metrics and progress tracking</div>
        </div>
      </CardContent>
    </Card>
  );
}