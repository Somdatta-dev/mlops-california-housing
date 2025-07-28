'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { TrainingControls } from './TrainingControls';
import { GPUMonitoringPanel } from './GPUMonitoringPanel';
import { TrainingProgressChart } from './TrainingProgressChart';
import { ModelComparisonTable } from './ModelComparisonTable';
import { HyperparameterTuning } from './HyperparameterTuning';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useApi } from '@/hooks/useApi';
import { 
  TrainingJob, 
  TrainingConfig, 
  GPUMetrics, 
  ModelComparison,
  HyperparameterPreset 
} from '@/types';

export function TrainingInterface() {
  const [activeJob, setActiveJob] = useState<TrainingJob | null>(null);
  const [gpuMetrics, setGpuMetrics] = useState<GPUMetrics | null>(null);
  const [modelComparisons, setModelComparisons] = useState<ModelComparison[]>([]);
  const [hyperparameterPresets, setHyperparameterPresets] = useState<HyperparameterPreset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<HyperparameterPreset | null>(null);

  const { isConnected } = useWebSocket();
  const { data: apiData, loading: apiLoading, error: apiError } = useApi(() => Promise.resolve({}));

  // Handle WebSocket data (disabled for now)
  useEffect(() => {
    // WebSocket handling would go here when implemented
  }, []);

  // Load initial data
  useEffect(() => {
    // Initialize with empty data for now
    setModelComparisons([]);
    setHyperparameterPresets([]);
  }, []);

  const loadModelComparisons = async () => {
    try {
      // Mock data for now since API endpoints don't exist
      setModelComparisons([]);
    } catch (error) {
      console.error('Failed to load model comparisons:', error);
    }
  };

  const loadHyperparameterPresets = async () => {
    try {
      // Mock data for now since API endpoints don't exist
      setHyperparameterPresets([]);
    } catch (error) {
      console.error('Failed to load hyperparameter presets:', error);
    }
  };

  const handleStartTraining = async (config: TrainingConfig) => {
    try {
      // Mock training start for now since API endpoints don't exist
      console.log('Starting training with config:', config);
      setActiveJob({
        id: 'mock-job-1',
        model_type: config.model_type,
        status: 'training',
        progress: 0,
        config,
      });
    } catch (error) {
      console.error('Failed to start training:', error);
    }
  };

  const handlePauseTraining = async () => {
    if (!activeJob) return;
    
    try {
      // Mock pause training for now
      setActiveJob({ ...activeJob, status: 'paused' });
    } catch (error) {
      console.error('Failed to pause training:', error);
    }
  };

  const handleStopTraining = async () => {
    if (!activeJob) return;
    
    try {
      // Mock stop training for now
      setActiveJob({ ...activeJob, status: 'completed' });
      await loadModelComparisons(); // Refresh model comparisons
    } catch (error) {
      console.error('Failed to stop training:', error);
    }
  };

  const handleResumeTraining = async () => {
    if (!activeJob) return;
    
    try {
      // Mock resume training for now
      setActiveJob({ ...activeJob, status: 'training' });
    } catch (error) {
      console.error('Failed to resume training:', error);
    }
  };

  const handleSelectModel = async (modelId: string) => {
    try {
      // Mock model selection for now
      console.log('Selecting model:', modelId);
      await loadModelComparisons(); // Refresh to show updated selection
    } catch (error) {
      console.error('Failed to select model:', error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Connection Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            Training Status
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          </CardTitle>
          <CardDescription>
            {isConnected ? 'Connected to training server' : 'Disconnected from training server'}
          </CardDescription>
        </CardHeader>
      </Card>

      <Tabs defaultValue="controls" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="controls">Training Controls</TabsTrigger>
          <TabsTrigger value="monitoring">GPU Monitoring</TabsTrigger>
          <TabsTrigger value="progress">Training Progress</TabsTrigger>
          <TabsTrigger value="comparison">Model Comparison</TabsTrigger>
        </TabsList>

        <TabsContent value="controls" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <TrainingControls
              activeJob={activeJob}
              onStartTraining={handleStartTraining}
              onPauseTraining={handlePauseTraining}
              onStopTraining={handleStopTraining}
              onResumeTraining={handleResumeTraining}
            />
            <HyperparameterTuning
              presets={hyperparameterPresets}
              selectedPreset={selectedPreset}
              onPresetSelect={setSelectedPreset}
              onCustomConfig={(config) => handleStartTraining(config)}
            />
          </div>
        </TabsContent>

        <TabsContent value="monitoring" className="space-y-6">
          <GPUMonitoringPanel 
            gpuMetrics={gpuMetrics}
            isConnected={isConnected}
          />
        </TabsContent>

        <TabsContent value="progress" className="space-y-6">
          <TrainingProgressChart 
            trainingJob={activeJob}
            isConnected={isConnected}
          />
        </TabsContent>

        <TabsContent value="comparison" className="space-y-6">
          <ModelComparisonTable
            models={modelComparisons}
            onSelectModel={handleSelectModel}
            onRefresh={loadModelComparisons}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}