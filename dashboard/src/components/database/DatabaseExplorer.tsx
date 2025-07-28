/**
 * Database Explorer Component
 * 
 * Main component for database exploration with filtering, search, pagination,
 * and data visualization capabilities.
 */

'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Database, TrendingUp, Download, Filter } from 'lucide-react';

import { PredictionTable } from './PredictionTable';
import { FilterPanel } from './FilterPanel';
import { ExportControls } from './ExportControls';
import { PredictionTrends } from './PredictionTrends';
import { DatabaseStats } from './DatabaseStats';
import { useDatabasePredictions, useDatabaseStats, usePredictionTrends, useDataExport } from '@/hooks/useApi';
import { FilterConfig, PaginationConfig } from '@/types';

export function DatabaseExplorer() {
  // State management
  const [filters, setFilters] = useState<FilterConfig>({});
  const [pagination, setPagination] = useState<PaginationConfig>({ page: 1, limit: 50 });
  const [activeTab, setActiveTab] = useState('data');

  // Prepare API parameters
  const apiParams = {
    page: pagination.page,
    limit: pagination.limit,
    model_version: filters.modelVersion,
    status: filters.status,
    start_date: filters.dateRange?.start,
    end_date: filters.dateRange?.end,
    search_term: filters.searchTerm,
  };

  const statsParams = {
    start_date: filters.dateRange?.start,
    end_date: filters.dateRange?.end,
  };

  // API hooks
  const { data: predictionsData, loading: predictionsLoading, error: predictionsError, refetch: refetchPredictions } = useDatabasePredictions(apiParams);
  const { data: statsData, loading: statsLoading, error: statsError, refetch: refetchStats } = useDatabaseStats(statsParams);
  const { data: trendsData, loading: trendsLoading, error: trendsError, refetch: refetchTrends } = usePredictionTrends({ days: 7, interval: 'hour' });
  const { exportData, loading: exportLoading } = useDataExport();

  // Handlers
  const handleFiltersChange = useCallback((newFilters: FilterConfig) => {
    setFilters(newFilters);
    setPagination(prev => ({ ...prev, page: 1 })); // Reset to first page when filters change
  }, []);

  const handlePaginationChange = useCallback((newPagination: PaginationConfig) => {
    setPagination(newPagination);
  }, []);

  const handleExport = useCallback(async (format: 'csv' | 'json') => {
    try {
      await exportData({
        format,
        model_version: filters.modelVersion,
        status: filters.status,
        start_date: filters.dateRange?.start,
        end_date: filters.dateRange?.end,
        search_term: filters.searchTerm,
        limit: 10000, // Export more records than displayed
      });
    } catch (error) {
      console.error('Export failed:', error);
    }
  }, [exportData, filters]);

  const handleRefresh = useCallback(() => {
    refetchPredictions();
    refetchStats();
    refetchTrends();
  }, [refetchPredictions, refetchStats, refetchTrends]);

  // Auto-refresh data every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      if (activeTab === 'data') {
        refetchPredictions();
      } else if (activeTab === 'stats') {
        refetchStats();
      } else if (activeTab === 'trends') {
        refetchTrends();
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [activeTab, refetchPredictions, refetchStats, refetchTrends]);

  return (
    <div className="space-y-6">
      {/* Header with Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Predictions</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {statsLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                statsData?.total_predictions?.toLocaleString() || '0'
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              All time predictions
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {statsLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                `${(statsData?.success_rate || 0).toFixed(1)}%`
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Successful predictions
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Processing Time</CardTitle>
            <Filter className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {statsLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                `${(statsData?.average_processing_time_ms || 0).toFixed(0)}ms`
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Average response time
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Failed Predictions</CardTitle>
            <Download className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {statsLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                statsData?.failed_predictions?.toLocaleString() || '0'
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Error count
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="data">Prediction Data</TabsTrigger>
          <TabsTrigger value="trends">Trends & Analytics</TabsTrigger>
          <TabsTrigger value="stats">Detailed Statistics</TabsTrigger>
        </TabsList>

        {/* Prediction Data Tab */}
        <TabsContent value="data" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Prediction History</CardTitle>
              <CardDescription>
                Browse and filter prediction records with advanced search capabilities
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Filter Panel */}
              <FilterPanel 
                filters={filters} 
                onFiltersChange={handleFiltersChange}
                onRefresh={handleRefresh}
              />

              {/* Export Controls */}
              <ExportControls 
                onExport={handleExport}
                loading={exportLoading}
                recordCount={predictionsData?.total_count || 0}
              />

              {/* Error Display */}
              {predictionsError && (
                <Alert variant="destructive">
                  <AlertDescription>
                    Failed to load prediction data: {predictionsError}
                  </AlertDescription>
                </Alert>
              )}

              {/* Prediction Table */}
              <PredictionTable
                data={predictionsData?.predictions || []}
                loading={predictionsLoading}
                pagination={{
                  page: pagination.page,
                  limit: pagination.limit,
                  total: predictionsData?.total_count || 0,
                }}
                onPaginationChange={handlePaginationChange}
                hasNext={predictionsData?.has_next || false}
                hasPrevious={predictionsData?.has_previous || false}
              />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Trends & Analytics Tab */}
        <TabsContent value="trends" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Prediction Trends & Analytics</CardTitle>
              <CardDescription>
                Visualize prediction patterns and trends over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              {trendsError && (
                <Alert variant="destructive" className="mb-4">
                  <AlertDescription>
                    Failed to load trends data: {trendsError}
                  </AlertDescription>
                </Alert>
              )}

              <PredictionTrends
                data={trendsData}
                loading={trendsLoading}
                onRefresh={refetchTrends}
              />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Detailed Statistics Tab */}
        <TabsContent value="stats" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Detailed Statistics</CardTitle>
              <CardDescription>
                Comprehensive statistics and performance metrics
              </CardDescription>
            </CardHeader>
            <CardContent>
              {statsError && (
                <Alert variant="destructive" className="mb-4">
                  <AlertDescription>
                    Failed to load statistics: {statsError}
                  </AlertDescription>
                </Alert>
              )}

              <DatabaseStats
                data={statsData}
                loading={statsLoading}
                onRefresh={refetchStats}
                filters={filters}
                onFiltersChange={handleFiltersChange}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}