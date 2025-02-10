import React, { useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Select } from "@/components/ui/select";

const PortfolioManager = () => {
  const [step, setStep] = useState(1);
  const [selectedAU, setSelectedAU] = useState('');
  const [range, setRange] = useState('');
  const [customerType, setCustomerType] = useState('all');
  const [portfolioAllocations, setPortfolioAllocations] = useState({});

  const mockData = {
    aus: ['AU1', 'AU2', 'AU3'],
    ranges: ['5km', '10km', '15km', '20km'],
    customerTypes: ['all', 'assigned', 'unassigned']
  };

  const renderStep = () => {
    switch(step) {
      case 1:
        return (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Select Administrative Unit</h3>
            <select 
              className="w-full p-2 border rounded"
              value={selectedAU}
              onChange={(e) => setSelectedAU(e.target.value)}
            >
              <option value="">Select AU</option>
              {mockData.aus.map(au => (
                <option key={au} value={au}>{au}</option>
              ))}
            </select>
          </div>
        );
      
      case 2:
        return (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Select Distance Range</h3>
            <select 
              className="w-full p-2 border rounded"
              value={range}
              onChange={(e) => setRange(e.target.value)}
            >
              <option value="">Select Range</option>
              {mockData.ranges.map(r => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </div>
        );

      case 3:
        return (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Customer Selection</h3>
            <select 
              className="w-full p-2 border rounded"
              value={customerType}
              onChange={(e) => setCustomerType(e.target.value)}
            >
              {mockData.customerTypes.map(type => (
                <option key={type} value={type}>
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </option>
              ))}
            </select>
            <div className="mt-4 p-4 bg-gray-100 rounded">
              <h4 className="font-medium">Customer Statistics</h4>
              <p>Assigned: 150</p>
              <p>Unassigned: 75</p>
            </div>
          </div>
        );

      case 4:
        return (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Portfolio Allocation</h3>
            <div className="space-y-2">
              <div className="flex justify-between p-2 bg-gray-100 rounded">
                <span>Portfolio A (100 customers)</span>
                <input 
                  type="number" 
                  className="w-20 p-1 border rounded"
                  placeholder="0"
                />
              </div>
              <div className="flex justify-between p-2 bg-gray-100 rounded">
                <span>Portfolio B (75 customers)</span>
                <input 
                  type="number" 
                  className="w-20 p-1 border rounded"
                  placeholder="0"
                />
              </div>
            </div>
          </div>
        );

      case 5:
        return (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Portfolio Visualization</h3>
            <div className="h-64 bg-gray-200 rounded flex items-center justify-center">
              Map Visualization
            </div>
          </div>
        );
    }
  };

  return (
    <Card className="w-full max-w-2xl">
      <CardHeader>
        <CardTitle>Portfolio Manager</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {renderStep()}
          <div className="flex justify-between mt-4">
            {step > 1 && (
              <button 
                className="px-4 py-2 bg-gray-200 rounded"
                onClick={() => setStep(step - 1)}
              >
                Previous
              </button>
            )}
            {step < 5 && (
              <button 
                className="px-4 py-2 bg-blue-500 text-white rounded ml-auto"
                onClick={() => setStep(step + 1)}
              >
                Next
              </button>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default PortfolioManager;
