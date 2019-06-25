package org.cloudbus.cloudsim;

import java.io.*;
import java.io.IOException;

/**
 * Instantiates a new utilization model for BitBrain cpuUtilizationData with variable cpuUtilizationData samples.
 *
 * @Author Shashikant Ilager
 */
public class UtilizationModelCPUBitBrainInMemory implements UtilizationModel {

    /** The scheduling interval. */
    private double schedulingInterval;

    /** The cpuUtilizationData (5 min * 288 = 24 hours). */
    private final double[] cpuUtilizationData;

    /**
     * Instantiates a new utilization model Bitbrain with variable cpuUtilizationData samples.
     *
     * @param inputPath the input path
     * @param dataSamples number of samples in the file
     * @throws NumberFormatException the number format exception
     * @throws IOException Signals that an I/O exception has occurred.
     */
    //TODO load other cpuUtilizationData fields and modify the cpuUtilizationData of bitbrains for 24 hour
    public UtilizationModelCPUBitBrainInMemory(String inputPath, double schedulingInterval, int dataSamples)
            throws NumberFormatException,
            IOException {
        setSchedulingInterval(schedulingInterval);
        cpuUtilizationData = new double[dataSamples]; // number of cpuUtilizationData sample entry it takes. Indicating how many samples you want

        // should be less than or equal to number of entry in cpuUtilizationData file
        BufferedReader bufferedReader = new BufferedReader(new FileReader(inputPath));
        int n = cpuUtilizationData.length;
        Log.printLine("Number of simulation intervals = "+ n);
        int count =0;
        String[] nextRow = null;
        for (int i = 0; i < n - 1; i++) {
            //Skip the CSV header
            String line= bufferedReader.readLine(); // TODO check null condition
            if (i==0)
                continue;
            if (line!=null)
                nextRow = line.split(";");
//            System.out.println("@@@@@@@@@@@@@@@@@@@@@@@@@@ nextrow: " + nextRow + " count: " + count++);

            cpuUtilizationData[i] = Double.parseDouble(nextRow[4]) / 100.0; // It was Intger.Valueof() - changed- March 20/19

//            Log.printDebugMessages("@ "+ UtilizationModelCPUBitBrainInMemory.class.getSimpleName() + " currentRow Data of Workload- Row: " + i +
//                    " Array size: " + nextRow.length +
//                    " Array  Data: " + Arrays.toString(nextRow));
//            Log.printLine("CPU Util Data: Cloudlet- " + i + " = " + cpuUtilizationData[i] + ((line == null) ? (" line is null -> copying prev value") : "" ));

        }
        cpuUtilizationData[n - 1] = cpuUtilizationData[n - 2];
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
            return cpuUtilizationData[(int) time / (int) getSchedulingInterval()];
        }
        int time1 = (int) Math.floor(time / getSchedulingInterval());
        int time2 = (int) Math.ceil(time / getSchedulingInterval());
        double utilization1 = cpuUtilizationData[time1];
        double utilization2 = cpuUtilizationData[time2];
        double delta = (utilization2 - utilization1) / ((time2 - time1) * getSchedulingInterval());
        double utilization = utilization1 + delta * (time - time1 * getSchedulingInterval());
        return utilization;
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
}
