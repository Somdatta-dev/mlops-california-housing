/**
 * Export Controls Component
 * 
 * Provides data export functionality with format selection
 * and download capabilities for CSV and JSON formats.
 */

'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Download, 
  FileText, 
  Database, 
  Loader2,
  CheckCircle,
  AlertCircle
} from 'lucide-react';

interface ExportControlsProps {
  onExport: (format: 'csv' | 'json') => Promise<void>;
  loading: boolean;
  recordCount: number;
}

export function ExportControls({ onExport, loading, recordCount }: ExportControlsProps) {
  const [selectedFormat, setSelectedFormat] = useState<'csv' | 'json'>('csv');
  const [exportStatus, setExportStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [exportMessage, setExportMessage] = useState<string>('');

  const handleExport = async () => {
    try {
      setExportStatus('idle');
      setExportMessage('');
      
      await onExport(selectedFormat);
      
      setExportStatus('success');
      setExportMessage(`Successfully exported ${recordCount.toLocaleString()} records as ${selectedFormat.toUpperCase()}`);
      
      // Clear success message after 5 seconds
      setTimeout(() => {
        setExportStatus('idle');
        setExportMessage('');
      }, 5000);
      
    } catch (error) {
      setExportStatus('error');
      setExportMessage(error instanceof Error ? error.message : 'Export failed');
      
      // Clear error message after 10 seconds
      setTimeout(() => {
        setExportStatus('idle');
        setExportMessage('');
      }, 10000);
    }
  };

  const getFormatIcon = (format: string) => {
    switch (format) {
      case 'csv':
        return <FileText className="h-4 w-4" />;
      case 'json':
        return <Database className="h-4 w-4" />;
      default:
        return <Download className="h-4 w-4" />;
    }
  };

  const getFormatDescription = (format: string) => {
    switch (format) {
      case 'csv':
        return 'Comma-separated values format, ideal for Excel and data analysis tools';
      case 'json':
        return 'JavaScript Object Notation format, ideal for programmatic processing';
      default:
        return '';
    }
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center space-x-2">
          <Download className="h-5 w-5" />
          <span>Export Data</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600">Format:</span>
              <Select
                value={selectedFormat}
                onValueChange={(value: 'csv' | 'json') => setSelectedFormat(value)}
                disabled={loading}
              >
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="csv">
                    <div className="flex items-center space-x-2">
                      <FileText className="h-4 w-4" />
                      <span>CSV</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="json">
                    <div className="flex items-center space-x-2">
                      <Database className="h-4 w-4" />
                      <span>JSON</span>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600">Records:</span>
              <Badge variant="secondary">
                {recordCount.toLocaleString()}
              </Badge>
            </div>
          </div>

          <Button
            onClick={handleExport}
            disabled={loading || recordCount === 0}
            className="flex items-center space-x-2"
          >
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              getFormatIcon(selectedFormat)
            )}
            <span>
              {loading ? 'Exporting...' : `Export ${selectedFormat.toUpperCase()}`}
            </span>
          </Button>
        </div>

        {/* Format Description */}
        <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded">
          <div className="flex items-start space-x-2">
            {getFormatIcon(selectedFormat)}
            <div>
              <div className="font-medium mb-1">{selectedFormat.toUpperCase()} Format</div>
              <div>{getFormatDescription(selectedFormat)}</div>
            </div>
          </div>
        </div>

        {/* Export Status Messages */}
        {exportStatus === 'success' && exportMessage && (
          <Alert className="border-green-200 bg-green-50">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <AlertDescription className="text-green-800">
              {exportMessage}
            </AlertDescription>
          </Alert>
        )}

        {exportStatus === 'error' && exportMessage && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              {exportMessage}
            </AlertDescription>
          </Alert>
        )}

        {/* Export Information */}
        <div className="text-xs text-gray-500 space-y-1">
          <div>• Export includes all filtered records up to 10,000 items</div>
          <div>• Files are automatically downloaded to your default download folder</div>
          <div>• CSV format includes flattened feature columns for easy analysis</div>
          <div>• JSON format preserves the complete data structure</div>
        </div>
      </CardContent>
    </Card>
  );
}