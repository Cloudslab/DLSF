package org.cloudbus.cloudsim;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class UtilizationModelRamBitBrainInMemory implements UtilizationModel {

    /** The scheduling interval. */
    private double schedulingInterval;

    /* * The cpuUtilizationData for memory utilization  */
    private final  double[] memoryUtilizationData;
    private final  double[] memoryProvisionedData;

    /**
     * Instantiates a new utilization model Bitbrain with variable cpuUtilizationData samples.
     *
     * @param inputPath the input path
     * @param dataSamples number of samples in the file
     * @throws NumberFormatException the number format exception
     * @throws IOException Signals that an I/O exception has occurred.
     */

    //TODO load other cpuUtilizationData fields and modify the cpuUtilizationData of bitbrains for 24 hour
    public UtilizationModelRamBitBrainInMemory (String inputPath, double schedulingInterval, int dataSamples)
            throws NumberFormatException,
            IOException {
        setSchedulingInterval(schedulingInterval);

        memoryUtilizationData = new double[dataSamples];
        memoryProvisionedData = new double[dataSamples];

        // should be less than or equal to number of entry in cpuUtilizationData file
        BufferedReader bufferedReader = new BufferedReader(new FileReader(inputPath));
        int n = memoryUtilizationData.length;
        String[] nextRow = null;

        for (int i = 0; i < n - 1; i++) {
            //Skip the CSV header
            String line= bufferedReader.readLine(); // TODO check null condition
            if (i==0)
                continue;
            if (line != null)
                nextRow = line.split(";");
            memoryUtilizationData[i] = Double.parseDouble(nextRow[6]);
            memoryProvisionedData[i] = Double.parseDouble(nextRow[5]);

//            Log.print("@ "+ UtilizationModelRamBitBrainInMemory.class.getSimpleName() + " currentRow Data of Workload- Row: " + i +
//                    " Array size: " + nextRow.length +
//                    " Array  Data: " + Arrays.toString(nextRow));
//            Log.printLine("Ram Util Data: Cloudlet- " + i + " = " + memoryUtilizationData[i] + ((line == null) ? (" line is null -> copying prev value") : "" ));

        }
        memoryUtilizationData[n - 1] = memoryUtilizationData[n - 2];
        bufferedReader.close();
    }

    /*
     * (non-Javadoc)
     * @see cloudsim.power.UtilizationModel#getUtilization(double)
     */
    // gives utilization percentage of machine/vm of that particular time
    @Override
    public double getUtilization(double time) {
        if (time % getSchedulingInterval() == 0) {
            return memoryUtilizationData[(int) time / (int) getSchedulingInterval()];
        }
        int time1 = (int) Math.floor(time / getSchedulingInterval());
        int time2 = (int) Math.ceil(time / getSchedulingInterval());
        double utilization1 = memoryUtilizationData[time1];
        double utilization2 = memoryUtilizationData[time2];


        double delta = (utilization2 - utilization1) / ((time2 - time1) * getSchedulingInterval());
        double utilization = utilization1 + delta * (time - time1 * getSchedulingInterval());

//        The utilization is in kB, we want to return the value in the range 0-1. Hence we take ratio between provisioned and used in KB.
        double provsioned1 = memoryProvisionedData[time1];
        double provisioned2 = memoryProvisionedData[time2];


        double proviosnedDelta= (provisioned2 - provsioned1) / ((time2 - time1) * getSchedulingInterval());
        double provisionedMemory = provsioned1 + proviosnedDelta * (time - time1 * getSchedulingInterval());

        double utilizationPercentage =0;
        // Bitbrain data set has some entry with 0 value, to avoid Math Exception as Divide by 0. In such case return value 0
        if (provisionedMemory!=0) {
            utilizationPercentage = utilization / provisionedMemory;
        }

        return  utilizationPercentage;

    }

    /**
     * Sets the scheduling interval.
     *
     * @param schedulingInterval the new scheduling interval
     */
    public void setSchedulingInterval(double schedulingInterval) {
        this.schedulingInterval = schedulingInterval;
    }

    /**
     * Gets the scheduling interval.
     *
     * @return the scheduling interval
     */
    public double getSchedulingInterval() {
        return schedulingInterval;
    }

//    @Override
//    public void setUtilization(double utilization, double time) {
//        // TODO Auto-generated method stub
//
//    }
}
