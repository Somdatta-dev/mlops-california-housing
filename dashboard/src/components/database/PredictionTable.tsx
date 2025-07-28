/**
 * Prediction Table Component
 * 
 * Displays prediction records in a table format with pagination,
 * sorting, and detailed view capabilities.
 */

'use client';

import { useState } from 'react';
import { format } from 'date-fns';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  ChevronLeft, 
  ChevronRight, 
  Eye, 
  CheckCircle, 
  XCircle, 
  Clock,
  Database
} from 'lucide-react';

import { DatabasePredictionRecord, PaginationConfig } from '@/types';

interface PredictionTableProps {
  data: DatabasePredictionRecord[];
  loading: boolean;
  pagination: PaginationConfig & { total: number };
  onPaginationChange: (pagination: PaginationConfig) => void;
  hasNext: boolean;
  hasPrevious: boolean;
}

export function PredictionTable({
  data,
  loading,
  pagination,
  onPaginationChange,
  hasNext,
  hasPrevious,
}: PredictionTableProps) {
  const [selectedRecord, setSelectedRecord] = useState<DatabasePredictionRecord | null>(null);

  const handlePageChange = (newPage: number) => {
    onPaginationChange({
      ...pagination,
      page: newPage,
    });
  };

  const handleLimitChange = (newLimit: number) => {
    onPaginationChange({
      page: 1,
      limit: newLimit,
    });
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'success':
        return (
          <Badge variant="default" className="bg-green-100 text-green-800 hover:bg-green-200">
            <CheckCircle className="w-3 h-3 mr-1" />
            Success
          </Badge>
        );
      case 'error':
        return (
          <Badge variant="destructive">
            <XCircle className="w-3 h-3 mr-1" />
            Error
          </Badge>
        );
      default:
        return (
          <Badge variant="secondary">
            <Clock className="w-3 h-3 mr-1" />
            {status}
          </Badge>
        );
    }
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      return format(new Date(timestamp), 'MMM dd, yyyy HH:mm:ss');
    } catch {
      return timestamp;
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value * 100000); // Convert to actual price range
  };

  const totalPages = Math.ceil(pagination.total / pagination.limit);

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-24" />
        </div>
        <div className="border rounded-lg">
          <Table>
            <TableHeader>
              <TableRow>
                {Array.from({ length: 8 }).map((_, i) => (
                  <TableHead key={i}>
                    <Skeleton className="h-4 w-20" />
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {Array.from({ length: 5 }).map((_, i) => (
                <TableRow key={i}>
                  {Array.from({ length: 8 }).map((_, j) => (
                    <TableCell key={j}>
                      <Skeleton className="h-4 w-16" />
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center justify-center py-12">
          <div className="text-center">
            <div className="text-gray-400 mb-4">
              <Database className="h-12 w-12 mx-auto" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No predictions found</h3>
            <p className="text-gray-500">
              Try adjusting your filters or check back later for new prediction data.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Table Header Info */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-gray-600">
          Showing {((pagination.page - 1) * pagination.limit) + 1} to{' '}
          {Math.min(pagination.page * pagination.limit, pagination.total)} of{' '}
          {pagination.total.toLocaleString()} predictions
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-600">Rows per page:</span>
          <select
            value={pagination.limit}
            onChange={(e) => handleLimitChange(Number(e.target.value))}
            className="border border-gray-300 rounded px-2 py-1 text-sm"
          >
            <option value={25}>25</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
            <option value={200}>200</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="border rounded-lg overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Request ID</TableHead>
              <TableHead>Timestamp</TableHead>
              <TableHead>Model</TableHead>
              <TableHead>Prediction</TableHead>
              <TableHead>Processing Time</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Batch ID</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.map((record) => (
              <TableRow key={record.id} className="hover:bg-gray-50">
                <TableCell className="font-mono text-sm">
                  {record.request_id.substring(0, 8)}...
                </TableCell>
                <TableCell className="text-sm">
                  {formatTimestamp(record.timestamp)}
                </TableCell>
                <TableCell>
                  <div className="flex flex-col">
                    <span className="text-sm font-medium">{record.model_version}</span>
                    <span className="text-xs text-gray-500">{record.model_stage}</span>
                  </div>
                </TableCell>
                <TableCell>
                  <div className="flex flex-col">
                    <span className="font-medium">{formatCurrency(record.prediction)}</span>
                    {record.confidence_lower && record.confidence_upper && (
                      <span className="text-xs text-gray-500">
                        ±{formatCurrency(Math.abs(record.confidence_upper - record.confidence_lower) / 2)}
                      </span>
                    )}
                  </div>
                </TableCell>
                <TableCell className="text-sm">
                  {record.processing_time_ms.toFixed(1)}ms
                </TableCell>
                <TableCell>
                  {getStatusBadge(record.status)}
                </TableCell>
                <TableCell className="text-sm">
                  {record.batch_id ? (
                    <span className="font-mono">{record.batch_id.substring(0, 8)}...</span>
                  ) : (
                    <span className="text-gray-400">-</span>
                  )}
                </TableCell>
                <TableCell>
                  <Dialog>
                    <DialogTrigger asChild>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedRecord(record)}
                      >
                        <Eye className="h-4 w-4" />
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
                      <DialogHeader>
                        <DialogTitle>Prediction Details</DialogTitle>
                        <DialogDescription>
                          Detailed information for prediction {record.request_id}
                        </DialogDescription>
                      </DialogHeader>
                      {selectedRecord && (
                        <div className="space-y-4">
                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <h4 className="font-medium mb-2">Basic Information</h4>
                              <div className="space-y-2 text-sm">
                                <div><strong>Request ID:</strong> {selectedRecord.request_id}</div>
                                <div><strong>Timestamp:</strong> {formatTimestamp(selectedRecord.timestamp)}</div>
                                <div><strong>Model Version:</strong> {selectedRecord.model_version}</div>
                                <div><strong>Model Stage:</strong> {selectedRecord.model_stage}</div>
                                <div><strong>Status:</strong> {getStatusBadge(selectedRecord.status)}</div>
                                <div><strong>Processing Time:</strong> {selectedRecord.processing_time_ms.toFixed(2)}ms</div>
                              </div>
                            </div>
                            <div>
                              <h4 className="font-medium mb-2">Prediction Results</h4>
                              <div className="space-y-2 text-sm">
                                <div><strong>Prediction:</strong> {formatCurrency(selectedRecord.prediction)}</div>
                                {selectedRecord.confidence_lower && (
                                  <div><strong>Confidence Lower:</strong> {formatCurrency(selectedRecord.confidence_lower)}</div>
                                )}
                                {selectedRecord.confidence_upper && (
                                  <div><strong>Confidence Upper:</strong> {formatCurrency(selectedRecord.confidence_upper)}</div>
                                )}
                                {selectedRecord.confidence_score && (
                                  <div><strong>Confidence Score:</strong> {(selectedRecord.confidence_score * 100).toFixed(1)}%</div>
                                )}
                              </div>
                            </div>
                          </div>
                          
                          <div>
                            <h4 className="font-medium mb-2">Input Features</h4>
                            <div className="grid grid-cols-2 gap-2 text-sm bg-gray-50 p-3 rounded">
                              {Object.entries(selectedRecord.input_features).map(([key, value]) => (
                                <div key={key}>
                                  <strong>{key}:</strong> {typeof value === 'number' ? value.toFixed(4) : value}
                                </div>
                              ))}
                            </div>
                          </div>

                          {selectedRecord.error_message && (
                            <div>
                              <h4 className="font-medium mb-2 text-red-600">Error Details</h4>
                              <div className="bg-red-50 p-3 rounded text-sm text-red-800">
                                {selectedRecord.error_message}
                              </div>
                            </div>
                          )}

                          <div>
                            <h4 className="font-medium mb-2">Request Metadata</h4>
                            <div className="space-y-2 text-sm">
                              {selectedRecord.batch_id && (
                                <div><strong>Batch ID:</strong> {selectedRecord.batch_id}</div>
                              )}
                              {selectedRecord.ip_address && (
                                <div><strong>IP Address:</strong> {selectedRecord.ip_address}</div>
                              )}
                              {selectedRecord.user_agent && (
                                <div><strong>User Agent:</strong> {selectedRecord.user_agent}</div>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </DialogContent>
                  </Dialog>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePageChange(pagination.page - 1)}
            disabled={!hasPrevious}
          >
            <ChevronLeft className="h-4 w-4 mr-1" />
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePageChange(pagination.page + 1)}
            disabled={!hasNext}
          >
            Next
            <ChevronRight className="h-4 w-4 ml-1" />
          </Button>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-600">
            Page {pagination.page} of {totalPages}
          </span>
        </div>
      </div>
    </div>
  );
}