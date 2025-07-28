'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { 
  Settings, 
  RotateCcw, 
  Play, 
  BookOpen,
  Lightbulb
} from 'lucide-react';
import { HyperparameterPreset, TrainingConfig } from '@/types';

interface HyperparameterTuningProps {
  presets: HyperparameterPreset[];
  selectedPreset: HyperparameterPreset | null;
  onPresetSelect: (preset: HyperparameterPreset) => void;
  onCustomConfig: (config: TrainingConfig) => void;
}

export function HyperparameterTuning({
  presets,
  selectedPreset,
  onPresetSelect,
  onCustomConfig,
}: HyperparameterTuningProps) {
  const [customConfig, setCustomConfig] = useState<TrainingConfig>({
    model_type: 'xgboost',
    hyperparameters: {
      max_depth: 6,
      n_estimators: 1000,
      learning_rate: 0.1,
      subsample: 0.8,
      epochs: 100,
      dropout_rate: 0.2,
      num_leaves: 31,
      feature_fraction: 0.8,
    },
    gpu_enabled: true,
    mixed_precision: true,
    early_stopping: true,
    validation_split: 0.2,
  });

  // Helper function to safely get numeric values from params
  const getNumericParam = (key: string, defaultValue: number): number => {
    const value = customConfig.hyperparameters[key];
    return typeof value === 'number' ? value : defaultValue;
  };

  const [activeTab, setActiveTab] = useState('presets');

  // Update custom config when preset is selected
  useEffect(() => {
    if (selectedPreset) {
      setCustomConfig({
        model_type: selectedPreset.model_type as any,
        hyperparameters: { ...selectedPreset.parameters },
        gpu_enabled: true,
        mixed_precision: true,
        early_stopping: true,
        validation_split: 0.2,
      });
    }
  }, [selectedPreset]);

  const getModelTypePresets = (modelType: string) => {
    return presets.filter(preset => preset.model_type === modelType);
  };

  const updateHyperparameter = (key: string, value: any) => {
    setCustomConfig(prev => ({
      ...prev,
      hyperparameters: {
        ...prev.hyperparameters,
        [key]: value,
      },
    }));
  };

  const resetToDefaults = () => {
    const defaultParams = getDefaultHyperparameters(customConfig.model_type);
    setCustomConfig(prev => ({
      ...prev,
      hyperparameters: defaultParams,
    }));
  };

  const getDefaultHyperparameters = (modelType: string) => {
    switch (modelType) {
      case 'xgboost':
        return {
          max_depth: 6,
          n_estimators: 1000,
          learning_rate: 0.1,
          subsample: 0.8,
          colsample_bytree: 0.8,
          tree_method: 'gpu_hist',
          gpu_id: 0,
        };
      case 'neural_network':
        return {
          hidden_layers: [512, 256, 128],
          batch_size: 1024,
          epochs: 100,
          learning_rate: 0.001,
          dropout_rate: 0.2,
          optimizer: 'adam',
        };
      case 'lightgbm':
        return {
          num_leaves: 31,
          n_estimators: 1000,
          learning_rate: 0.1,
          feature_fraction: 0.8,
          bagging_fraction: 0.8,
          device: 'gpu',
        };
      case 'random_forest':
        return {
          n_estimators: 1000,
          max_depth: 10,
          min_samples_split: 2,
          min_samples_leaf: 1,
          max_features: 'sqrt',
        };
      case 'linear_regression':
        return {
          fit_intercept: true,
          normalize: false,
          alpha: 1.0,
        };
      default:
        return {};
    }
  };

  const renderHyperparameterControls = () => {
    const params = customConfig.hyperparameters;
    const modelType = customConfig.model_type;

    switch (modelType) {
      case 'xgboost':
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Max Depth</Label>
                <Slider
                  value={[getNumericParam('max_depth', 6)]}
                  onValueChange={([value]) => updateHyperparameter('max_depth', value)}
                  min={3}
                  max={15}
                  step={1}
                />
                <div className="text-sm text-gray-500">{getNumericParam('max_depth', 6)}</div>
              </div>
              <div className="space-y-2">
                <Label>N Estimators</Label>
                <Slider
                  value={[getNumericParam('n_estimators', 1000)]}
                  onValueChange={([value]) => updateHyperparameter('n_estimators', value)}
                  min={100}
                  max={5000}
                  step={100}
                />
                <div className="text-sm text-gray-500">{getNumericParam('n_estimators', 1000)}</div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Learning Rate</Label>
                <Slider
                  value={[getNumericParam('learning_rate', 0.1)]}
                  onValueChange={([value]) => updateHyperparameter('learning_rate', value)}
                  min={0.01}
                  max={0.3}
                  step={0.01}
                />
                <div className="text-sm text-gray-500">{getNumericParam('learning_rate', 0.1)}</div>
              </div>
              <div className="space-y-2">
                <Label>Subsample</Label>
                <Slider
                  value={[getNumericParam('subsample', 0.8)]}
                  onValueChange={([value]) => updateHyperparameter('subsample', value)}
                  min={0.5}
                  max={1.0}
                  step={0.1}
                />
                <div className="text-sm text-gray-500">{getNumericParam('subsample', 0.8)}</div>
              </div>
            </div>
          </div>
        );

      case 'neural_network':
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Hidden Layers (comma-separated)</Label>
              <Input
                value={Array.isArray(customConfig.hyperparameters.hidden_layers) ? customConfig.hyperparameters.hidden_layers.join(', ') : '512, 256, 128'}
                onChange={(e) => {
                  const layers = e.target.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
                  updateHyperparameter('hidden_layers', layers);
                }}
                placeholder="512, 256, 128"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Batch Size</Label>
                <Select
                  value={String(getNumericParam('batch_size', 1024))}
                  onValueChange={(value) => updateHyperparameter('batch_size', parseInt(value))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="256">256</SelectItem>
                    <SelectItem value="512">512</SelectItem>
                    <SelectItem value="1024">1024</SelectItem>
                    <SelectItem value="2048">2048</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Epochs</Label>
                <Slider
                  value={[getNumericParam('epochs', 100)]}
                  onValueChange={([value]) => updateHyperparameter('epochs', value)}
                  min={10}
                  max={500}
                  step={10}
                />
                <div className="text-sm text-gray-500">{getNumericParam('epochs', 100)}</div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Learning Rate</Label>
                <Slider
                  value={[getNumericParam('learning_rate', 0.001)]}
                  onValueChange={([value]) => updateHyperparameter('learning_rate', value)}
                  min={0.0001}
                  max={0.01}
                  step={0.0001}
                />
                <div className="text-sm text-gray-500">{getNumericParam('learning_rate', 0.001)}</div>
              </div>
              <div className="space-y-2">
                <Label>Dropout Rate</Label>
                <Slider
                  value={[getNumericParam('dropout_rate', 0.2)]}
                  onValueChange={([value]) => updateHyperparameter('dropout_rate', value)}
                  min={0.0}
                  max={0.5}
                  step={0.1}
                />
                <div className="text-sm text-gray-500">{getNumericParam('dropout_rate', 0.2)}</div>
              </div>
            </div>
          </div>
        );

      case 'lightgbm':
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Num Leaves</Label>
                <Slider
                  value={[getNumericParam('num_leaves', 31)]}
                  onValueChange={([value]) => updateHyperparameter('num_leaves', value)}
                  min={10}
                  max={100}
                  step={1}
                />
                <div className="text-sm text-gray-500">{getNumericParam('num_leaves', 31)}</div>
              </div>
              <div className="space-y-2">
                <Label>N Estimators</Label>
                <Slider
                  value={[getNumericParam('n_estimators', 1000)]}
                  onValueChange={([value]) => updateHyperparameter('n_estimators', value)}
                  min={100}
                  max={5000}
                  step={100}
                />
                <div className="text-sm text-gray-500">{getNumericParam('n_estimators', 1000)}</div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Learning Rate</Label>
                <Slider
                  value={[getNumericParam('learning_rate', 0.1)]}
                  onValueChange={([value]) => updateHyperparameter('learning_rate', value)}
                  min={0.01}
                  max={0.3}
                  step={0.01}
                />
                <div className="text-sm text-gray-500">{getNumericParam('learning_rate', 0.1)}</div>
              </div>
              <div className="space-y-2">
                <Label>Feature Fraction</Label>
                <Slider
                  value={[getNumericParam('feature_fraction', 0.8)]}
                  onValueChange={([value]) => updateHyperparameter('feature_fraction', value)}
                  min={0.5}
                  max={1.0}
                  step={0.1}
                />
                <div className="text-sm text-gray-500">{getNumericParam('feature_fraction', 0.8)}</div>
              </div>
            </div>
          </div>
        );

      default:
        return (
          <div className="text-center py-8 text-gray-500">
            Select a model type to configure hyperparameters
          </div>
        );
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="h-5 w-5" />
          Hyperparameter Tuning
        </CardTitle>
        <CardDescription>
          Configure model hyperparameters using presets or custom settings
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="presets">Presets</TabsTrigger>
            <TabsTrigger value="custom">Custom</TabsTrigger>
          </TabsList>

          <TabsContent value="presets" className="space-y-4">
            <div className="space-y-2">
              <Label>Model Type</Label>
              <Select
                value={customConfig.model_type}
                onValueChange={(value) => setCustomConfig(prev => ({ ...prev, model_type: value as any }))}
              >
                <SelectTrigger>
                  <SelectValue />
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

            <div className="space-y-3">
              <Label>Available Presets</Label>
              {getModelTypePresets(customConfig.model_type).length > 0 ? (
                <div className="space-y-2">
                  {getModelTypePresets(customConfig.model_type).map((preset) => (
                    <div
                      key={preset.id}
                      className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                        selectedPreset?.id === preset.id
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => onPresetSelect(preset)}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium">{preset.name}</div>
                          <div className="text-sm text-gray-500">{preset.description}</div>
                        </div>
                        <div className="flex items-center gap-2">
                          {preset.recommended_for.map((tag) => (
                            <Badge key={tag} variant="secondary" className="text-xs">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-6 text-gray-500">
                  <BookOpen className="h-8 w-8 mx-auto mb-2" />
                  <p>No presets available for {customConfig.model_type.replace('_', ' ')}</p>
                  <p className="text-sm">Switch to Custom tab to configure manually</p>
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="custom" className="space-y-4">
            <div className="space-y-2">
              <Label>Model Type</Label>
              <Select
                value={customConfig.model_type}
                onValueChange={(value) => {
                  const newModelType = value as any;
                  setCustomConfig(prev => ({
                    ...prev,
                    model_type: newModelType,
                    hyperparameters: getDefaultHyperparameters(newModelType),
                  }));
                }}
              >
                <SelectTrigger>
                  <SelectValue />
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

            <Separator />

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label>Hyperparameters</Label>
                <Button variant="outline" size="sm" onClick={resetToDefaults}>
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Reset to Defaults
                </Button>
              </div>
              {renderHyperparameterControls()}
            </div>

            <Separator />

            <div className="space-y-4">
              <Label>Training Options</Label>
              <div className="grid grid-cols-2 gap-4">
                <div className="flex items-center justify-between">
                  <Label htmlFor="gpu-enabled">GPU Acceleration</Label>
                  <Switch
                    id="gpu-enabled"
                    checked={customConfig.gpu_enabled}
                    onCheckedChange={(checked) => 
                      setCustomConfig(prev => ({ ...prev, gpu_enabled: checked }))
                    }
                  />
                </div>
                <div className="flex items-center justify-between">
                  <Label htmlFor="mixed-precision">Mixed Precision</Label>
                  <Switch
                    id="mixed-precision"
                    checked={customConfig.mixed_precision}
                    onCheckedChange={(checked) => 
                      setCustomConfig(prev => ({ ...prev, mixed_precision: checked }))
                    }
                  />
                </div>
                <div className="flex items-center justify-between">
                  <Label htmlFor="early-stopping">Early Stopping</Label>
                  <Switch
                    id="early-stopping"
                    checked={customConfig.early_stopping}
                    onCheckedChange={(checked) => 
                      setCustomConfig(prev => ({ ...prev, early_stopping: checked }))
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label>Validation Split</Label>
                  <Slider
                    value={[customConfig.validation_split]}
                    onValueChange={([value]) => 
                      setCustomConfig(prev => ({ ...prev, validation_split: value }))
                    }
                    min={0.1}
                    max={0.4}
                    step={0.05}
                  />
                  <div className="text-sm text-gray-500">{customConfig.validation_split}</div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>

        <Separator className="my-6" />

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <Lightbulb className="h-4 w-4" />
            <span>
              {customConfig.gpu_enabled ? 'GPU acceleration enabled' : 'CPU training only'}
              {customConfig.mixed_precision && ' • Mixed precision training'}
              {customConfig.early_stopping && ' • Early stopping enabled'}
            </span>
          </div>
          <Button onClick={() => onCustomConfig(customConfig)} className="flex items-center gap-2">
            <Play className="h-4 w-4" />
            Start Training
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}