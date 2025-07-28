import { TrainingInterface } from "@/components/training/TrainingInterface";

export default function TrainingPage() {
  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Model Training</h1>
        <p className="text-gray-600 mt-2">
          Train and monitor GPU-accelerated machine learning models with real-time performance tracking.
        </p>
      </div>
      <TrainingInterface />
    </div>
  );
}