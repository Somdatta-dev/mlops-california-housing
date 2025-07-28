/**
 * Interactive Prediction Form Component
 * Provides input validation and user feedback for housing predictions
 */

'use client';

import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { 
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';

import { 
  Home, 
  DollarSign, 
  Users, 
  MapPin, 
  Calendar,
  Loader2,
  AlertCircle,
  CheckCircle,
  Info
} from 'lucide-react';

import { usePrediction } from '@/hooks/useApi';
import { formatNumber } from '@/utils/format';
import type { PredictionRequest, ModelInfo } from '@/types';

// Validation schema for California Housing data
const predictionSchema = z.object({
  MedInc: z.number()
    .min(0, 'Median income must be positive')
    .max(15, 'Median income seems too high (max: 15)')
    .refine(val => val > 0, 'Median income is required'),
  
  HouseAge: z.number()
    .min(1, 'House age must be at least 1 year')
    .max(52, 'House age cannot exceed 52 years')
    .int('House age must be a whole number'),
  
  AveRooms: z.number()
    .min(1, 'Average rooms must be at least 1')
    .max(20, 'Average rooms seems too high (max: 20)')
    .refine(val => val >= 1, 'Average rooms is required'),
  
  AveBedrms: z.number()
    .min(0, 'Average bedrooms cannot be negative')
    .max(5, 'Average bedrooms seems too high (max: 5)')
    .refine(val => val >= 0, 'Average bedrooms is required'),
  
  Population: z.number()
    .min(3, 'Population must be at least 3')
    .max(35682, 'Population seems too high (max: 35,682)')
    .int('Population must be a whole number'),
  
  AveOccup: z.number()
    .min(0.5, 'Average occupancy must be at least 0.5')
    .max(1243, 'Average occupancy seems too high (max: 1,243)')
    .refine(val => val >= 0.5, 'Average occupancy is required'),
  
  Latitude: z.number()
    .min(32.54, 'Latitude must be within California bounds (min: 32.54)')
    .max(41.95, 'Latitude must be within California bounds (max: 41.95)'),
  
  Longitude: z.number()
    .min(-124.35, 'Longitude must be within California bounds (min: -124.35)')
    .max(-114.31, 'Longitude must be within California bounds (max: -114.31)')
});

type PredictionFormData = z.infer<typeof predictionSchema>;

interface PredictionFormProps {
  onSubmit: (data: PredictionRequest) => void;
  isConnected: boolean;
  modelInfo?: ModelInfo | null;
}

export function PredictionForm({ onSubmit, isConnected, modelInfo }: PredictionFormProps) {
  const { predict, loading, error, data: predictionResult } = usePrediction();
  const [showAdvanced, setShowAdvanced] = useState(false);

  const form = useForm<PredictionFormData>({
    resolver: zodResolver(predictionSchema),
    defaultValues: {
      MedInc: 5.0,
      HouseAge: 10,
      AveRooms: 6.0,
      AveBedrms: 1.2,
      Population: 3000,
      AveOccup: 3.0,
      Latitude: 34.0,
      Longitude: -118.0
    }
  });

  const handleSubmit = async (data: PredictionFormData) => {
    try {
      await predict(data);
      onSubmit(data);
    } catch (err) {
      console.error('Prediction failed:', err);
    }
  };

  const loadSampleData = (sampleType: 'low' | 'medium' | 'high') => {
    const samples = {
      low: {
        MedInc: 2.5,
        HouseAge: 35,
        AveRooms: 4.5,
        AveBedrms: 1.1,
        Population: 5000,
        AveOccup: 4.2,
        Latitude: 33.8,
        Longitude: -117.9
      },
      medium: {
        MedInc: 5.0,
        HouseAge: 15,
        AveRooms: 6.0,
        AveBedrms: 1.2,
        Population: 3000,
        AveOccup: 3.0,
        Latitude: 34.0,
        Longitude: -118.0
      },
      high: {
        MedInc: 8.5,
        HouseAge: 5,
        AveRooms: 7.5,
        AveBedrms: 1.3,
        Population: 2000,
        AveOccup: 2.5,
        Latitude: 37.8,
        Longitude: -122.4
      }
    };

    form.reset(samples[sampleType]);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Home className="h-5 w-5" />
          <span>Housing Prediction</span>
        </CardTitle>
        <CardDescription>
          Enter housing characteristics to get a price prediction
        </CardDescription>
        
        {/* Model Status */}
        {modelInfo && (
          <div className="flex items-center space-x-2 pt-2">
            <Badge variant={modelInfo.gpu_accelerated ? 'default' : 'secondary'}>
              {modelInfo.model_type}
            </Badge>
            <Badge variant="outline">
              v{modelInfo.model_version}
            </Badge>
            {modelInfo.gpu_accelerated && (
              <Badge variant="default" className="bg-green-600">
                GPU Accelerated
              </Badge>
            )}
          </div>
        )}
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Connection Status Alert */}
        {!isConnected && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              WebSocket disconnected. Predictions may not appear in real-time feed.
            </AlertDescription>
          </Alert>
        )}

        {/* Sample Data Buttons */}
        <div className="flex space-x-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => loadSampleData('low')}
          >
            Low Value Sample
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => loadSampleData('medium')}
          >
            Medium Value Sample
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => loadSampleData('high')}
          >
            High Value Sample
          </Button>
        </div>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-4">
            {/* Basic Information */}
            <div className="grid gap-4 md:grid-cols-2">
              <FormField
                control={form.control}
                name="MedInc"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="flex items-center space-x-1">
                      <DollarSign className="h-4 w-4" />
                      <span>Median Income</span>
                    </FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        step="0.1"
                        placeholder="5.0"
                        {...field}
                        onChange={e => field.onChange(parseFloat(e.target.value) || 0)}
                      />
                    </FormControl>
                    <FormDescription>
                      Median income in block group (in tens of thousands)
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="HouseAge"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="flex items-center space-x-1">
                      <Calendar className="h-4 w-4" />
                      <span>House Age</span>
                    </FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        placeholder="10"
                        {...field}
                        onChange={e => field.onChange(parseInt(e.target.value) || 0)}
                      />
                    </FormControl>
                    <FormDescription>
                      Median house age in block group (years)
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            {/* Room Information */}
            <div className="grid gap-4 md:grid-cols-2">
              <FormField
                control={form.control}
                name="AveRooms"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="flex items-center space-x-1">
                      <Home className="h-4 w-4" />
                      <span>Average Rooms</span>
                    </FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        step="0.1"
                        placeholder="6.0"
                        {...field}
                        onChange={e => field.onChange(parseFloat(e.target.value) || 0)}
                      />
                    </FormControl>
                    <FormDescription>
                      Average number of rooms per household
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="AveBedrms"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Average Bedrooms</FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        step="0.1"
                        placeholder="1.2"
                        {...field}
                        onChange={e => field.onChange(parseFloat(e.target.value) || 0)}
                      />
                    </FormControl>
                    <FormDescription>
                      Average number of bedrooms per household
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            {/* Population Information */}
            <div className="grid gap-4 md:grid-cols-2">
              <FormField
                control={form.control}
                name="Population"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="flex items-center space-x-1">
                      <Users className="h-4 w-4" />
                      <span>Population</span>
                    </FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        placeholder="3000"
                        {...field}
                        onChange={e => field.onChange(parseInt(e.target.value) || 0)}
                      />
                    </FormControl>
                    <FormDescription>
                      Block group population
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="AveOccup"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Average Occupancy</FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        step="0.1"
                        placeholder="3.0"
                        {...field}
                        onChange={e => field.onChange(parseFloat(e.target.value) || 0)}
                      />
                    </FormControl>
                    <FormDescription>
                      Average number of household members
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            {/* Location Information */}
            <div>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="mb-2"
              >
                <MapPin className="h-4 w-4 mr-1" />
                {showAdvanced ? 'Hide' : 'Show'} Location Details
              </Button>

              {showAdvanced && (
                <div className="grid gap-4 md:grid-cols-2 p-4 border rounded-lg">
                  <FormField
                    control={form.control}
                    name="Latitude"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Latitude</FormLabel>
                        <FormControl>
                          <Input
                            type="number"
                            step="0.01"
                            placeholder="34.0"
                            {...field}
                            onChange={e => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </FormControl>
                        <FormDescription>
                          Block group latitude (32.54 to 41.95)
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="Longitude"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Longitude</FormLabel>
                        <FormControl>
                          <Input
                            type="number"
                            step="0.01"
                            placeholder="-118.0"
                            {...field}
                            onChange={e => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </FormControl>
                        <FormDescription>
                          Block group longitude (-124.35 to -114.31)
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
              )}
            </div>

            {/* Submit Button */}
            <Button
              type="submit"
              className="w-full"
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Predicting...
                </>
              ) : (
                <>
                  <DollarSign className="h-4 w-4 mr-2" />
                  Get Price Prediction
                </>
              )}
            </Button>
          </form>
        </Form>

        {/* Error Display */}
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Prediction failed: {error}
            </AlertDescription>
          </Alert>
        )}

        {/* Success Display */}
        {predictionResult && (
          <Alert>
            <CheckCircle className="h-4 w-4" />
            <AlertDescription>
              <div className="space-y-1">
                <div className="font-medium">
                  Predicted House Value: ${formatNumber(predictionResult.prediction / 100000, 1)}K
                </div>
                <div className="text-sm text-muted-foreground">
                  Processing time: {formatNumber(predictionResult.processing_time_ms, 0)}ms
                  {predictionResult.confidence_interval && (
                    <span className="ml-2">
                      • Confidence: ${formatNumber(predictionResult.confidence_interval[0] / 100000, 1)}K - ${formatNumber(predictionResult.confidence_interval[1] / 100000, 1)}K
                    </span>
                  )}
                </div>
              </div>
            </AlertDescription>
          </Alert>
        )}

        {/* Info */}
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            This model predicts California housing prices based on 1990 census data. 
            Values are estimates and should not be used for actual real estate decisions.
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
}