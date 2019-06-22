package org.cloudbus.cloudsim.examples.power.DeepRL;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.VmAllocationPolicy;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.examples.power.Constants;
import org.cloudbus.cloudsim.examples.power.Helper;
import org.cloudbus.cloudsim.examples.power.RunnerAbstract;
import org.cloudbus.cloudsim.power.*;

import java.io.IOException;
import java.util.Calendar;
import java.util.List;

/**
 * @author Shreshth Tuli
 * @since June 16, 2019
 */

public class DeepRLRunner extends RunnerAbstract {

    /**
     * @param enableOutput enable output or not
     * @param outputToFile output to file or not
     * @param inputFolder path of input folder
     * @param outputFolder path of output folder
     * @param workload workload name
     * @param vmAllocationPolicy allocation policy name
     * @param vmSelectionPolicy selection policy name
     * @param parameter parameter for running tests
     */
    public DeepRLRunner(
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

    @Override
    protected void init(String inputFolder) {
        try {
            CloudSim.init(1, Calendar.getInstance(), false);

            broker = Helper.createBroker();
            int brokerId = broker.getId();

            // Data center creation at RunnerAbstract.start()
            cloudletList = DeepRLHelper.createCloudletListBitBrain(brokerId, inputFolder); //ThermalHelper.createCloudletList(brokerId, ThermalConstants.NUMBER_OF_VMS);
            vmList = DeepRLHelper.createVmList(brokerId, cloudletList.size());
            hostList = DeepRLHelper.createHostList(DeepRLConstants.NUMBER_OF_HOSTS);
        } catch (Exception e) {
            e.printStackTrace();
            Log.printLine("The simulation has been terminated due to an unexpected error");
            System.exit(0);
        }
    }

    /**
     * Starts the simulation.
     *
     * @param experimentName the experiment name
     * @param outputFolder the output folder
     * @param vmAllocationPolicy the vm allocation policy
     */
    @Override
    protected void start(String experimentName, String outputFolder, VmAllocationPolicy vmAllocationPolicy) {
        System.out.println("Starting " + experimentName);

        try {
            DRLDatacenter datacenter = (DRLDatacenter) Helper.createDatacenter(
                    "Datacenter",
                    DRLDatacenter.class,
                    hostList,
                    vmAllocationPolicy);

            datacenter.setDisableMigrations(false);

            broker.submitVmList(vmList);
            broker.submitCloudletList(cloudletList);

            CloudSim.terminateSimulation(Constants.SIMULATION_LIMIT);
            double lastClock = CloudSim.startSimulation();

            List<Cloudlet> newList = broker.getCloudletReceivedList();
            Log.printLine("Received " + newList.size() + " cloudlets");

            CloudSim.stopSimulation();

            Helper.printResults(
                    datacenter,
                    vmList,
                    lastClock,
                    experimentName,
                    Constants.OUTPUT_CSV,
                    outputFolder);

        } catch (Exception e) {
            e.printStackTrace();
            Log.printLine("The simulation has been terminated due to an unexpected error");
            System.exit(0);
        }

        Log.printLine("Finished " + experimentName);
    }

    /**
     * Gets the vm allocation policy.
     *
     * @param vmAllocationPolicyName the vm allocation policy name
     * @param vmSelectionPolicyName the vm selection policy name
     * @param parameterName the parameter name
     * @return the vm allocation policy
     */
    @Override
    protected VmAllocationPolicy getVmAllocationPolicy(
            String vmAllocationPolicyName,
            String vmSelectionPolicyName,
            String parameterName) {
        VmAllocationPolicy vmAllocationPolicy = null;
        PowerVmSelectionPolicy vmSelectionPolicy = null;
        if (!vmSelectionPolicyName.isEmpty()) {
            vmSelectionPolicy = getVmSelectionPolicy(vmSelectionPolicyName);
        }
        double parameter = 0;
        if (!parameterName.isEmpty()) {
            parameter = Double.valueOf(parameterName);
        }
        if (vmAllocationPolicyName.equals("iqr")) {
            PowerVmAllocationPolicyMigrationAbstract fallbackVmSelectionPolicy = new PowerVmAllocationPolicyMigrationStaticThreshold(
                    hostList,
                    vmSelectionPolicy,
                    0.7);
            vmAllocationPolicy = new PowerVmAllocationPolicyMigrationInterQuartileRange(
                    hostList,
                    vmSelectionPolicy,
                    parameter,
                    fallbackVmSelectionPolicy);
        } else if (vmAllocationPolicyName.equals("mad")) {
            PowerVmAllocationPolicyMigrationAbstract fallbackVmSelectionPolicy = new PowerVmAllocationPolicyMigrationStaticThreshold(
                    hostList,
                    vmSelectionPolicy,
                    0.7);
            vmAllocationPolicy = new PowerVmAllocationPolicyMigrationMedianAbsoluteDeviation(
                    hostList,
                    vmSelectionPolicy,
                    parameter,
                    fallbackVmSelectionPolicy);
        } else if (vmAllocationPolicyName.equals("lr")) {
            PowerVmAllocationPolicyMigrationAbstract fallbackVmSelectionPolicy = new PowerVmAllocationPolicyMigrationStaticThreshold(
                    hostList,
                    vmSelectionPolicy,
                    0.7);
            vmAllocationPolicy = new PowerVmAllocationPolicyMigrationLocalRegression(
                    hostList,
                    vmSelectionPolicy,
                    parameter,
                    Constants.SCHEDULING_INTERVAL,
                    fallbackVmSelectionPolicy);
        } else if (vmAllocationPolicyName.equals("lrr")) {
            PowerVmAllocationPolicyMigrationAbstract fallbackVmSelectionPolicy = new PowerVmAllocationPolicyMigrationStaticThreshold(
                    hostList,
                    vmSelectionPolicy,
                    0.7);
            vmAllocationPolicy = new PowerVmAllocationPolicyMigrationLocalRegressionRobust(
                    hostList,
                    vmSelectionPolicy,
                    parameter,
                    Constants.SCHEDULING_INTERVAL,
                    fallbackVmSelectionPolicy);
        } else if (vmAllocationPolicyName.equals("thr")) {
            vmAllocationPolicy = new PowerVmAllocationPolicyMigrationStaticThreshold(
                    hostList,
                    vmSelectionPolicy,
                    parameter);
        } else if (vmAllocationPolicyName.equals("dvfs")) {
            vmAllocationPolicy = new PowerVmAllocationPolicySimple(hostList);
        } else if(vmAllocationPolicyName.equals("deepRL-alloc")){
            PowerVmAllocationPolicyMigrationAbstract fallbackVmSelectionPolicy = new PowerVmAllocationPolicyMigrationStaticThreshold(
                    hostList,
                    vmSelectionPolicy,
                    0.7);
            vmAllocationPolicy = new DRLVmAllocationPolicy(hostList, fallbackVmSelectionPolicy);
        } else {
            System.out.println("Unknown VM allocation policy: " + vmAllocationPolicyName);
            System.exit(0);
        }
        return vmAllocationPolicy;
    }

    /**
     * Gets the vm selection policy.
     *
     * @param vmSelectionPolicyName the vm selection policy name
     * @return the vm selection policy
     */
    @Override
    protected PowerVmSelectionPolicy getVmSelectionPolicy(String vmSelectionPolicyName) {
        PowerVmSelectionPolicy vmSelectionPolicy = null;
        if (vmSelectionPolicyName.equals("mc")) {
            vmSelectionPolicy = new PowerVmSelectionPolicyMaximumCorrelation(
                    new PowerVmSelectionPolicyMinimumMigrationTime());
        } else if (vmSelectionPolicyName.equals("mmt")) {
            vmSelectionPolicy = new PowerVmSelectionPolicyMinimumMigrationTime();
        } else if (vmSelectionPolicyName.equals("mu")) {
            vmSelectionPolicy = new PowerVmSelectionPolicyMinimumUtilization();
        } else if (vmSelectionPolicyName.equals("rs")) {
            vmSelectionPolicy = new PowerVmSelectionPolicyRandomSelection();
        } else if(vmSelectionPolicyName.equals("deepRL-sel")){
            vmSelectionPolicy = new DRLVmSelectionPolicy();
        } else {
            System.out.println("Unknown VM selection policy: " + vmSelectionPolicyName);
            System.exit(0);
        }
        return vmSelectionPolicy;
    }

    public static void main(String[] args) throws IOException {
        boolean enableOutput = true;
        boolean outputToFile = false;
        String inputFolder = "";
        String outputFolder = "";
        String workload = "random"; // Random workload
        String vmAllocationPolicy = "lr"; // Local Regression (LR) VM allocation policy
        String vmSelectionPolicy = "mmt"; // Minimum Migration Time (MMT) VM selection policy
        String parameter = "1.2"; // the safety parameter of the LR policy

        new DeepRLRunner(
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
