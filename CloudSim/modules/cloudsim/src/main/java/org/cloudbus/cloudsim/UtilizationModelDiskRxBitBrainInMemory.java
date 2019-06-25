package org.cloudbus.cloudsim;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class UtilizationModelDiskRxBitBrainInMemory implements UtilizationModel{

    /** The scheduling interval. */
    private double schedulingInterval;

    /**
     * The cpuUtilizationData for Incoming diskBandwidth/Received Utilization
     */
    private  final  double[] diskBandwidthRXUtilizationData; // Received

    private  final  double [] diskBandwidthTXUtilizationData; // Transmitted

    /**
     * Instantiates a new utilization model Bitbrain with variable cpuUtilizationData samples.
     *
     * @param inputPath the input path
     * @param dataSamples number of samples in the file
     * @throws NumberFormatException the number format exception
     * @throws IOException Signals that an I/O exception has occurred.
     */

    //TODO load other cpuUtilizationData fields and modify the cpuUtilizationData of bitbrains for 24 hour
    public UtilizationModelDiskRxBitBrainInMemory(String inputPath, double schedulingInterval, int dataSamples)
            throws NumberFormatException,
            IOException {
        setSchedulingInterval(schedulingInterval);
        diskBandwidthRXUtilizationData = new double[dataSamples];
        diskBandwidthTXUtilizationData = new double[dataSamples];


        // should be less than or equal to number of entry in cpuUtilizationData file
        BufferedReader bufferedReader = new BufferedReader(new FileReader(inputPath));
        int n = diskBandwidthRXUtilizationData.length;
        String[] nextRow = null;

        for (int i = 0; i < n - 1; i++) {
            //Skip the CSV header
            String line= bufferedReader.readLine(); // TODO check null condition
            if (i==0)
                continue;
            if(line != null)
                nextRow = line.split(";");
            diskBandwidthRXUtilizationData[i] =  Double.parseDouble(nextRow[7]);
            diskBandwidthTXUtilizationData[i] = Double.parseDouble(nextRow[8]);
//            Log.print("@ "+ UtilizationModelNetworkRxBitBrainInMemory.class.getSimpleName() + " currentRow Data of Workload- Row: " + i +
//                    " Array size: " + nextRow.length +
//                    " Array  Data: " + Arrays.toString(nextRow));
//            Log.printLine("DiskRx Util Data: Cloudlet- " + i + " = " + diskBandwidthRXUtilizationData[i] +
//                    " Transmitted: " + diskBandwidthTXUtilizationData[i]);

        }
        diskBandwidthRXUtilizationData[n - 1] = diskBandwidthRXUtilizationData[n - 2];
        diskBandwidthTXUtilizationData[n-1] = diskBandwidthTXUtilizationData[n-2];
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
            return diskBandwidthRXUtilizationData[(int) time / (int) getSchedulingInterval()];
        }
        int time1 = (int) Math.floor(time / getSchedulingInterval());
        int time2 = (int) Math.ceil(time / getSchedulingInterval());

        // Total diskBandwidth usage is combination of Rx and TX (Incoming and transmitting)
        double utilization1 = diskBandwidthRXUtilizationData[time1] + diskBandwidthTXUtilizationData[time1];
        double utilization2 = diskBandwidthRXUtilizationData[time2] + diskBandwidthTXUtilizationData[time2];
        double delta = (utilization2 - utilization1) / ((time2 - time1) * getSchedulingInterval());
        double utilization = utilization1 + delta * (time - time1 * getSchedulingInterval());

        double provisionediskBandwidth = 100000; // 100 Mbit/s
        double utilizationPercentage =0;
        //@Author Shash- Send back utilization ratio 0-1  //
        if (provisionediskBandwidth!=0) {
            // Bitbrain usage has units in KB/s and Vm has unit of Mbit
//            1 megabit = 10002 bits
//            1 megabit = 10002 × (1/8000) kilobytes
//            1 megabit = (1000/8) kilobytes
//            1 Mbit = 125 KB
            utilizationPercentage = utilization / provisionediskBandwidth*125;
        }

        return  utilizationPercentage; // 0-1
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
