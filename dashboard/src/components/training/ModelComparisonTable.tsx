'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from '@/components/ui/table';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { 
  Trophy, 
  RefreshCw, 
  Download, 
  Search, 
  Filter,
  CheckCircle,
  Clock,
  XCircle,
  Zap,
  TrendingUp,
  TrendingDown
} from 'lucide-react';
import { ModelComparison } from '@/types';

interface ModelComparisonTableProps {
  models: ModelComparison[];
  onSelectModel: (modelId: string) => void;
  onRefresh: () => void;
}

export function ModelComparisonTable({ 
  models, 
  onSelectModel, 
  onRefresh 
}: ModelComparisonTableProps) {
  const [sortBy, setSortBy] = useState<'rmse' | 'mae' | 'r2' | 'training_time'>('rmse');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [filterModelType, setFilterModelType] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'training':
        return <Clock className="h-4 w-4 text-blue-600" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-600" />;
      case 'selected':
        return <Trophy className="h-4 w-4 text-yellow-600" />;
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'training':
        return 'bg-blue-100 text-blue-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'selected':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const filteredAndSortedModels = models
    .filter(model => {
      if (filterStatus !== 'all' && model.status !== filterStatus) return false;
      if (filterModelType !== 'all' && model.model_type !== filterModelType) return false;
      if (searchTerm && !model.model_version.toLowerCase().includes(searchTerm.toLowerCase())) return false;
      return true;
    })
    .sort((a, b) => {
      const aValue = a.performance_metrics[sortBy];
      const bValue = b.performance_metrics[sortBy];
      
      if (sortOrder === 'asc') {
        return aValue - bValue;
      } else {
        return bValue - aValue;
      }
    });

  const getBestModel = (metric: 'rmse' | 'mae' | 'r2') => {
    if (models.length === 0) return null;
    
    return models.reduce((best, current) => {
      const currentValue = current.performance_metrics[metric];
      const bestValue = best.performance_metrics[metric];
      
      // For RMSE and MAE, lower is better. For R2, higher is better.
      if (metric === 'r2') {
        return currentValue > bestValue ? current : best;
      } else {
        return currentValue < bestValue ? current : best;
      }
    });
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  const exportData = () => {
    const csvContent = [
      ['Model ID', 'Model Type', 'Version', 'RMSE', 'MAE', 'R²', 'Training Time', 'GPU Accelerated', 'Status', 'Training Date'].join(','),
      ...filteredAndSortedModels.map(model => [
        model.model_id,
        model.model_type,
        model.model_version,
        model.performance_metrics.rmse.toFixed(6),
        model.performance_metrics.mae.toFixed(6),
        model.performance_metrics.r2.toFixed(6),
        model.performance_metrics.training_time.toFixed(2),
        model.gpu_accelerated ? 'Yes' : 'No',
        model.status,
        model.training_date
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model_comparison.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const bestRMSE = getBestModel('rmse');
  const bestMAE = getBestModel('mae');
  const bestR2 = getBestModel('r2');

  return (
    <div className="space-y-6">
      {/* Performance Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingDown className="h-4 w-4 text-green-600" />
              Best RMSE
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1">
              <div className="text-2xl font-bold">
                {bestRMSE?.performance_metrics.rmse.toFixed(6) || 'N/A'}
              </div>
              <div className="text-sm text-gray-500">
                {bestRMSE?.model_type.replace('_', ' ').toUpperCase() || 'No models'}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingDown className="h-4 w-4 text-blue-600" />
              Best MAE
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1">
              <div className="text-2xl font-bold">
                {bestMAE?.performance_metrics.mae.toFixed(6) || 'N/A'}
              </div>
              <div className="text-sm text-gray-500">
                {bestMAE?.model_type.replace('_', ' ').toUpperCase() || 'No models'}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-purple-600" />
              Best R²
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1">
              <div className="text-2xl font-bold">
                {bestR2?.performance_metrics.r2.toFixed(6) || 'N/A'}
              </div>
              <div className="text-sm text-gray-500">
                {bestR2?.model_type.replace('_', ' ').toUpperCase() || 'No models'}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Model Comparison Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Trophy className="h-5 w-5" />
                Model Comparison
              </CardTitle>
              <CardDescription>
                Compare performance metrics across all trained models
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={onRefresh}>
                <RefreshCw className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="sm" onClick={exportData}>
                <Download className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Filters */}
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <Search className="h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search models..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-48"
              />
            </div>
            <Select value={filterStatus} onValueChange={setFilterStatus}>
              <SelectTrigger className="w-32">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="training">Training</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
                <SelectItem value="selected">Selected</SelectItem>
              </SelectContent>
            </Select>
            <Select value={filterModelType} onValueChange={setFilterModelType}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Model Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="xgboost">XGBoost</SelectItem>
                <SelectItem value="neural_network">Neural Network</SelectItem>
                <SelectItem value="lightgbm">LightGBM</SelectItem>
                <SelectItem value="random_forest">Random Forest</SelectItem>
                <SelectItem value="linear_regression">Linear Regression</SelectItem>
              </SelectContent>
            </Select>
            <Select value={sortBy} onValueChange={(value: typeof sortBy) => setSortBy(value)}>
              <SelectTrigger className="w-32">
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="rmse">RMSE</SelectItem>
                <SelectItem value="mae">MAE</SelectItem>
                <SelectItem value="r2">R²</SelectItem>
                <SelectItem value="training_time">Training Time</SelectItem>
              </SelectContent>
            </Select>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
            >
              {sortOrder === 'asc' ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
            </Button>
          </div>

          {/* Table */}
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Status</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Version</TableHead>
                  <TableHead className="text-right">RMSE</TableHead>
                  <TableHead className="text-right">MAE</TableHead>
                  <TableHead className="text-right">R²</TableHead>
                  <TableHead className="text-right">Training Time</TableHead>
                  <TableHead>GPU</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredAndSortedModels.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={10} className="text-center py-8">
                      <div className="text-gray-500">
                        <Filter className="h-8 w-8 mx-auto mb-2" />
                        No models match the current filters
                      </div>
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredAndSortedModels.map((model) => (
                    <TableRow key={model.model_id}>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          {getStatusIcon(model.status)}
                          <Badge className={getStatusColor(model.status)}>
                            {model.status}
                          </Badge>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="font-medium">
                          {model.model_type.replace('_', ' ').toUpperCase()}
                        </div>
                      </TableCell>
                      <TableCell>
                        <code className="text-sm bg-gray-100 px-2 py-1 rounded">
                          {model.model_version}
                        </code>
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {model.performance_metrics.rmse.toFixed(6)}
                        {model === bestRMSE && (
                          <Trophy className="h-3 w-3 text-yellow-500 inline ml-1" />
                        )}
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {model.performance_metrics.mae.toFixed(6)}
                        {model === bestMAE && (
                          <Trophy className="h-3 w-3 text-yellow-500 inline ml-1" />
                        )}
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {model.performance_metrics.r2.toFixed(6)}
                        {model === bestR2 && (
                          <Trophy className="h-3 w-3 text-yellow-500 inline ml-1" />
                        )}
                      </TableCell>
                      <TableCell className="text-right">
                        {formatDuration(model.performance_metrics.training_time)}
                      </TableCell>
                      <TableCell>
                        {model.gpu_accelerated ? (
                          <Zap className="h-4 w-4 text-yellow-500" />
                        ) : (
                          <span className="text-gray-400">CPU</span>
                        )}
                      </TableCell>
                      <TableCell className="text-sm text-gray-500">
                        {new Date(model.training_date).toLocaleDateString()}
                      </TableCell>
                      <TableCell>
                        {model.status === 'completed' && (
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => onSelectModel(model.model_id)}
                          >
                            Select
                          </Button>
                        )}
                        {model.status === 'selected' && (
                          <Badge variant="default">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            Active
                          </Badge>
                        )}
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>

          {/* Summary */}
          <div className="text-sm text-gray-500">
            Showing {filteredAndSortedModels.length} of {models.length} models
          </div>
        </CardContent>
      </Card>
    </div>
  );
}