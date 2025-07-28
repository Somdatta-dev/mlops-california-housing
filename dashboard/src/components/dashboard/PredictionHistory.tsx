/**
 * Prediction History Component
 * Displays historical predictions with filtering, search, and pagination
 */

'use client';

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Search, 
  Filter, 
  Download, 
  Calendar,
  SortAsc,
  SortDesc,
  RefreshCw,
  FileText,
  Database,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';

import { usePredictionHistory } from '@/hooks/useApi';
import { formatNumber, formatDuration, formatDate, formatRelativeTime } from '@/utils/format';
import type { PredictionResponse, FilterConfig, PaginationConfig } from '@/types';

interface SortConfig {
  key: keyof PredictionResponse | 'none';
  direction: 'asc' | 'desc';
}

export function PredictionHistory() {
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'timestamp', direction: 'desc' });
  const [filters, setFilters] = useState<FilterConfig>({});

  const { data: historyData, loading, error, refetch } = usePredictionHistory(
    pageSize,
    (currentPage - 1) * pageSize
  );

  // Filter and sort data
  const processedData = useMemo(() => {
    if (!historyData) return [];

    let filtered = [...historyData];

    // Apply search filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(item => 
        item.request_id.toLowerCase().includes(term) ||
        item.model_version.toLowerCase().includes(term) ||
        item.prediction.toString().includes(term)
      );
    }

    // Apply date range filter
    if (filters.dateRange && filters.dateRange.start && filters.dateRange.end) {
      const startDate = new Date(filters.dateRange.start);
      const endDate = new Date(filters.dateRange.end);
      filtered = filtered.filter(item => {
        const itemDate = new Date(item.timestamp);
        return itemDate >= startDate && itemDate <= endDate;
      });
    }

    // Apply model version filter
    if (filters.modelVersion) {
      filtered = filtered.filter(item => item.model_version === filters.modelVersion);
    }

    // Apply sorting
    if (sortConfig.key !== 'none') {
      filtered.sort((a, b) => {
        const aVal = a[sortConfig.key as keyof PredictionResponse];
        const bVal = b[sortConfig.key as keyof PredictionResponse];
        
        let comparison = 0;
        if (aVal !== undefined && bVal !== undefined) {
          if (aVal < bVal) comparison = -1;
          if (aVal > bVal) comparison = 1;
        }
        
        return sortConfig.direction === 'desc' ? -comparison : comparison;
      });
    }

    return filtered;
  }, [historyData, searchTerm, filters, sortConfig]);

  // Get unique model versions for filter
  const modelVersions = useMemo(() => {
    if (!historyData) return [];
    return [...new Set(historyData.map(item => item.model_version))];
  }, [historyData]);

  const handleSort = (key: keyof PredictionResponse) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  const handleExport = (format: 'csv' | 'json') => {
    if (!processedData.length) return;

    let content: string;
    let filename: string;
    let mimeType: string;

    if (format === 'csv') {
      const headers = ['Request ID', 'Prediction', 'Model Version', 'Processing Time (ms)', 'Timestamp'];
      const rows = processedData.map(item => [
        item.request_id,
        item.prediction.toString(),
        item.model_version,
        item.processing_time_ms.toString(),
        item.timestamp
      ]);
      
      content = [headers, ...rows].map(row => row.join(',')).join('\n');
      filename = `predictions_${new Date().toISOString().split('T')[0]}.csv`;
      mimeType = 'text/csv';
    } else {
      content = JSON.stringify(processedData, null, 2);
      filename = `predictions_${new Date().toISOString().split('T')[0]}.json`;
      mimeType = 'application/json';
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const SortButton = ({ column, children }: { column: keyof PredictionResponse; children: React.ReactNode }) => (
    <Button
      variant="ghost"
      size="sm"
      className="h-8 px-2"
      onClick={() => handleSort(column)}
    >
      {children}
      {sortConfig.key === column && (
        sortConfig.direction === 'asc' ? 
          <SortAsc className="ml-1 h-3 w-3" /> : 
          <SortDesc className="ml-1 h-3 w-3" />
      )}
    </Button>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-medium">Prediction History</h3>
          <p className="text-sm text-muted-foreground">
            Browse and analyze historical predictions
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            disabled={loading}
          >
            <RefreshCw className={`h-4 w-4 mr-1 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleExport('csv')}
            disabled={!processedData.length}
          >
            <Download className="h-4 w-4 mr-1" />
            Export CSV
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleExport('json')}
            disabled={!processedData.length}
          >
            <FileText className="h-4 w-4 mr-1" />
            Export JSON
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center">
            <Filter className="h-4 w-4 mr-2" />
            Filters & Search
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {/* Search */}
            <div className="space-y-2">
              <Label htmlFor="search">Search</Label>
              <div className="relative">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  id="search"
                  placeholder="Search by ID, model, or value..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-8"
                />
              </div>
            </div>

            {/* Model Version Filter */}
            <div className="space-y-2">
              <Label>Model Version</Label>
              <Select
                value={filters.modelVersion || ''}
                onValueChange={(value) => 
                  setFilters(prev => ({ 
                    ...prev, 
                    modelVersion: value || undefined 
                  }))
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="All versions" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">All versions</SelectItem>
                  {modelVersions.map(version => (
                    <SelectItem key={version} value={version}>
                      {version}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Page Size */}
            <div className="space-y-2">
              <Label>Items per page</Label>
              <Select
                value={pageSize.toString()}
                onValueChange={(value) => {
                  setPageSize(parseInt(value));
                  setCurrentPage(1);
                }}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="10">10</SelectItem>
                  <SelectItem value="20">20</SelectItem>
                  <SelectItem value="50">50</SelectItem>
                  <SelectItem value="100">100</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Clear Filters */}
            <div className="space-y-2">
              <Label>&nbsp;</Label>
              <Button
                variant="outline"
                className="w-full"
                onClick={() => {
                  setSearchTerm('');
                  setFilters({});
                  setSortConfig({ key: 'timestamp', direction: 'desc' });
                }}
              >
                Clear Filters
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base flex items-center">
                <Database className="h-4 w-4 mr-2" />
                Results
              </CardTitle>
              <CardDescription>
                {loading ? 'Loading...' : `${processedData.length} predictions found`}
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertDescription>
                Failed to load prediction history: {error}
              </AlertDescription>
            </Alert>
          )}

          {loading ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="h-6 w-6 animate-spin mr-2" />
              Loading predictions...
            </div>
          ) : processedData.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Database className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <h3 className="text-lg font-medium mb-2">No predictions found</h3>
              <p className="text-sm">
                {searchTerm || Object.keys(filters).length > 0
                  ? 'Try adjusting your filters or search terms'
                  : 'No prediction history available yet'
                }
              </p>
            </div>
          ) : (
            <>
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>
                        <SortButton column="request_id">Request ID</SortButton>
                      </TableHead>
                      <TableHead>
                        <SortButton column="prediction">Prediction</SortButton>
                      </TableHead>
                      <TableHead>
                        <SortButton column="model_version">Model</SortButton>
                      </TableHead>
                      <TableHead>
                        <SortButton column="processing_time_ms">Processing Time</SortButton>
                      </TableHead>
                      <TableHead>
                        <SortButton column="timestamp">Timestamp</SortButton>
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {processedData.map((prediction) => (
                      <TableRow key={prediction.request_id}>
                        <TableCell className="font-mono text-xs">
                          {prediction.request_id.slice(-12)}
                        </TableCell>
                        <TableCell className="font-medium">
                          ${formatNumber(prediction.prediction / 100000, 1)}K
                          {prediction.confidence_interval && (
                            <div className="text-xs text-muted-foreground">
                              ±${formatNumber(
                                (prediction.confidence_interval[1] - prediction.confidence_interval[0]) / 200000,
                                1
                              )}K
                            </div>
                          )}
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline" className="text-xs">
                            {prediction.model_version}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          {formatDuration(prediction.processing_time_ms)}
                        </TableCell>
                        <TableCell>
                          <div className="text-sm">
                            {formatDate(prediction.timestamp)}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {formatRelativeTime(prediction.timestamp)}
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>

              {/* Pagination */}
              <div className="flex items-center justify-between mt-4">
                <div className="text-sm text-muted-foreground">
                  Showing {((currentPage - 1) * pageSize) + 1} to {Math.min(currentPage * pageSize, processedData.length)} of {processedData.length} results
                </div>
                <div className="flex items-center space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                    disabled={currentPage === 1}
                  >
                    <ChevronLeft className="h-4 w-4" />
                    Previous
                  </Button>
                  <span className="text-sm">
                    Page {currentPage} of {Math.ceil(processedData.length / pageSize)}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(prev => prev + 1)}
                    disabled={currentPage >= Math.ceil(processedData.length / pageSize)}
                  >
                    Next
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}