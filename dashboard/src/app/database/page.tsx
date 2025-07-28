/**
 * Database Explorer Page
 * 
 * This page provides comprehensive database exploration capabilities including:
 * - Browsing prediction history with filtering and search
 * - Pagination system for efficient data browsing
 * - Data export functionality (CSV, JSON)
 * - Data visualization components for prediction trends and patterns
 */

'use client';

import { DatabaseExplorer } from '@/components/database/DatabaseExplorer';

export default function DatabasePage() {
  return (
    <div className="container mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Database Explorer</h1>
        <p className="text-gray-600 mt-2">
          Browse prediction history, analyze trends, and export data with advanced filtering capabilities.
        </p>
      </div>
      
      <DatabaseExplorer />
    </div>
  );
}