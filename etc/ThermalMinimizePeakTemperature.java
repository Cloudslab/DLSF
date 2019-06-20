package com.cloudbus.cloudsim.examples.power.thermal;

import org.cloudbus.cloudsim.util.ExecutionTimeMeasurer;

import java.io.IOException;


public class ThermalMinimizePeakTemperature {

    /**
     * The main method.
     *
     * @param args the arguments
     * @throws IOException Signals that an I/O exception has occurred.
     */
    public static void main(String[] args) throws IOException {
        boolean enableOutput = true;  // true and flase to see on console detail.
        boolean outputToFile = true;
//        String inputFolder =  Thermal.class.getClassLoader().getResource("workload/planetlab").getPath();//"";
        String inputFolder =  ThermalMinimizePeakTemperature.class.getClassLoader().getResource("workload/bitbrain").getPath();//"";
        String outputFolder = "output";

        String workload = "fastStorage"; // "temp";//  "temp";//"random"; // Random workload //
        String vmAllocationPolicy = "thermalMinPeakTemp";//;"thermalCoolest";//  "thermalRandom";//;//"thermal";
        String vmSelectionPolicy = "mmt";//mu";
        String parameter = "200";


        new BitBrainRunner(
                enableOutput,
                outputToFile,
                inputFolder,
                outputFolder,
                workload,
                vmAllocationPolicy,
                vmSelectionPolicy,
                parameter);


    }


}
