package com.cloudbus.cloudsim.examples.power.thermal;

import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.examples.power.Helper;
import org.cloudbus.cloudsim.examples.power.RunnerAbstract;
import org.cloudbus.cloudsim.examples.power.planetlab.PlanetLabHelper;

import java.util.Calendar;

public class BitBrainRunner extends RunnerAbstract {

    /**
     * @param enableOutput
     * @param outputToFile
     * @param inputFolder
     * @param outputFolder
     * @param workload
     * @param vmAllocationPolicy
     * @param vmSelectionPolicy
     * @param parameter
     */
    public BitBrainRunner(
            boolean enableOutput,
            boolean outputToFile,
            String inputFolder,
            String outputFolder,
            String workload,
            String vmAllocationPolicy,
            String vmSelectionPolicy,
            String parameter) {
        super(
                enableOutput,
                outputToFile,
                inputFolder,
                outputFolder,
                workload,
                vmAllocationPolicy,
                vmSelectionPolicy,
                parameter);
    }

    /*
     * (non-Javadoc)
     *
     * @see org.cloudbus.cloudsim.examples.power.RunnerAbstract#init(java.lang.String)
     */
    @Override
    protected void init(String inputFolder) {
        try {
            CloudSim.init(1, Calendar.getInstance(), false);

            broker = Helper.createBroker();
            int brokerId = broker.getId();

            // Data center creation  at RunnerAbstract.start()
            cloudletList = BitBrainHelper.createCloudletListBitBrain(brokerId, inputFolder); //ThermalHelper.createCloudletList(brokerId, ThermalConstants.NUMBER_OF_VMS);
            vmList = BitBrainHelper.createVmList(brokerId, cloudletList.size());
            hostList = BitBrainHelper.createHostList(BitBrainConstants.NUMBER_OF_HOSTS);
        } catch (Exception e) {
            e.printStackTrace();
            Log.printLine("The simulation has been terminated due to an unexpected error");
            System.exit(0);
        }
    }

}
