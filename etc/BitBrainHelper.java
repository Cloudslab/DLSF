package com.cloudbus.cloudsim.examples.power.thermal;

import org.cloudbus.cloudsim.*;

import org.cloudbus.cloudsim.power.PowerHost;
import org.cloudbus.cloudsim.power.PowerHostUtilizationHistory;
import org.cloudbus.cloudsim.power.PowerVm;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

public class BitBrainHelper{
        /**
         * Creates the cloudlet list bitbrain dataset
         *
         * @param brokerId the broker id
         * @param inputFolderName the input folder name
         * @return the list
         * @throws FileNotFoundException the file not found exception
         */
        public static List<Cloudlet> createCloudletListBitBrain(int brokerId, String inputFolderName)
                throws FileNotFoundException {
            List<Cloudlet> list = new ArrayList<Cloudlet>();

            long fileSize = 300;
            long outputSize = 300;
            int datasamples = BitBrainConstants.NUMBER_OF_DATA_SAMPLES;
            Log.printLine("@ " + ThermalMinimizePeakTemperature.class.getSimpleName() + " inputFolfer: " + inputFolderName);
            UtilizationModel utilizationModelNull = new UtilizationModelNull();

            File inputFolder = new File(inputFolderName);
            File[] files = inputFolder.listFiles();
            Log.printLine("@ " + ThermalMinimizePeakTemperature.class.getSimpleName() + " inputFolfer: " + inputFolder + " Number of files: " + files.length);

            for (int i = 0; i < files.length; i++) {
                Cloudlet cloudlet = null;
                try {
                    System.out.println("@SSI- Filenumber- " + i + " filepath- " + files[i].getAbsolutePath() );
                    cloudlet = new Cloudlet(
                            i,
                            BitBrainConstants.CLOUDLET_LENGTH,
                            BitBrainConstants.CLOUDLET_PES,
                            fileSize,
                            outputSize,
                            new UtilizationModelCPUBitBrainInMemory(files[i].getAbsolutePath(),BitBrainConstants.SCHEDULING_INTERVAL, datasamples),
                            new UtilizationModelRamBitBrainInMemory(files[i].getAbsolutePath(),BitBrainConstants.SCHEDULING_INTERVAL, datasamples),
//                            utilizationModelNull,
                            new UtilizationModelNetworkRxBitBrainInMemory(files[i].getAbsolutePath(),BitBrainConstants.SCHEDULING_INTERVAL, datasamples)
//                            utilizationModelNull,
//                            files[i].getAbsolutePath()
                    );
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(0);
                }
                cloudlet.setUserId(brokerId);
                cloudlet.setVmId(i);
                list.add(cloudlet);
            }

            return list;
        }


    /**
     * Creates the vm list.
     *
     * @param brokerId the broker id
     * @param vmsNumber the vms number
     *
     * @return the list< vm>
     */
    public static List<Vm> createVmList(int brokerId, int vmsNumber) {
        List<Vm> vms = new ArrayList<Vm>();
        for (int i = 0; i < vmsNumber; i++) {
            int vmType = i / (int) Math.ceil((double) vmsNumber / BitBrainConstants.VM_TYPES);
            vms.add(new PowerVm(
                    i,
                    brokerId,
                    BitBrainConstants.VM_MIPS[vmType],
                    BitBrainConstants.VM_PES[vmType],
                    BitBrainConstants.VM_RAM[vmType],
                    BitBrainConstants.VM_BW,
                    BitBrainConstants.VM_SIZE,
                    1,
                    "Xen",
                    new CloudletSchedulerDynamicWorkload(BitBrainConstants.VM_MIPS[vmType], BitBrainConstants.VM_PES[vmType]),
                    BitBrainConstants.SCHEDULING_INTERVAL));
        }
        return vms;
    }

    /**
     * Creates the host list.
     *
     * @param hostsNumber the hosts number
     *
     * @return the list< power host>
     */
    public static List<PowerHost> createHostList(int hostsNumber) {
        List<PowerHost> hostList = new ArrayList<PowerHost>();
        for (int i = 0; i < hostsNumber; i++) {
//            int hostType = i % BitBrainConstants.HOST_TYPES;
               int hostType =0;


            List<Pe> peList = new ArrayList<Pe>();

            for (int j = 0; j < BitBrainConstants.HOST_PES[hostType]; j++) {
                peList.add(new Pe(j, new PeProvisionerSimple(BitBrainConstants.HOST_MIPS[hostType])));
            }

            hostList.add(new PowerHostUtilizationHistory(
                    i,
                    new RamProvisionerSimple(BitBrainConstants.HOST_RAM[hostType]),
                    new BwProvisionerSimple(BitBrainConstants.HOST_BW),
                    BitBrainConstants.HOST_STORAGE,
                    peList,
                    new VmSchedulerTimeSharedOverSubscription(peList),
                    BitBrainConstants.HOST_POWER[hostType]));
        }
        Log.printDebugMessages("Number of hosts created-> " + hostList.size());
        return hostList;
    }

}
