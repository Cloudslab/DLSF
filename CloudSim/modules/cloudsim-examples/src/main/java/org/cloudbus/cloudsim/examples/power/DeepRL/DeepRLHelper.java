package org.cloudbus.cloudsim.examples.power.DeepRL;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.UtilizationModel;
import org.cloudbus.cloudsim.UtilizationModelNull;
import org.cloudbus.cloudsim.*;

import org.cloudbus.cloudsim.cost.models.CostModelAzure.*;
import org.cloudbus.cloudsim.power.DRLHost;
import org.cloudbus.cloudsim.power.DRLVm;
import org.cloudbus.cloudsim.power.PowerHost;
import org.cloudbus.cloudsim.power.PowerVm;
import org.cloudbus.cloudsim.power.models.PowerModelSpecPowerDellPowerEdgeC6320;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Shreshth Tuli
 * @since June 16, 2019
 */

public class DeepRLHelper {

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
        int datasamples = DeepRLConstants.NUMBER_OF_DATA_SAMPLES;
        Log.printLine("@ " + ThermalMinimizePeakTemperature.class.getSimpleName() + " inputFolder: " + inputFolderName);
        UtilizationModel utilizationModelNull = new UtilizationModelNull();

        File inputFolder = new File(inputFolderName);
        File[] files = inputFolder.listFiles();
        Log.printLine("@ " + DeepRLRunner.class.getSimpleName() + " inputFolder: " + inputFolder + " Number of files: " + files.length);

        for (int i = 0; i < files.length; i++) {
            Cloudlet cloudlet = null;
            try {
                System.out.println("\n@SSI- Filenumber- " + i + " filepath- " + files[i].getAbsolutePath() );
                cloudlet = new Cloudlet(
                        i,
                        DeepRLConstants.CLOUDLET_LENGTH,
                        DeepRLConstants.CLOUDLET_PES,
                        fileSize,
                        outputSize,
                        new UtilizationModelCPUBitBrainInMemory(files[i].getAbsolutePath(),DeepRLConstants.SCHEDULING_INTERVAL, datasamples),
                        new UtilizationModelRamBitBrainInMemory(files[i].getAbsolutePath(),DeepRLConstants.SCHEDULING_INTERVAL, datasamples),
                        new UtilizationModelNetworkRxBitBrainInMemory(files[i].getAbsolutePath(),DeepRLConstants.SCHEDULING_INTERVAL, datasamples),
                        new UtilizationModelDiskRxBitBrainInMemeory(files[i].getAbsolutePath(),DeepRLConstants.SCHEDULING_INTERVAL, datasamples),
                        false
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
            int vmType = i / (int) Math.ceil((double) vmsNumber / DeepRLConstants.VM_TYPES);
            vms.add(new DRLVm(
                    i,
                    brokerId,
                    DeepRLConstants.VM_MIPS[vmType],
                    DeepRLConstants.VM_PES[vmType],
                    DeepRLConstants.VM_RAM[vmType],
                    DeepRLConstants.VM_BW,
                    DeepRLConstants.VM_DISKBW,
                    DeepRLConstants.VM_SIZE,
                    1,
                    "Xen",
                    new CloudletSchedulerDynamicWorkload(DeepRLConstants.VM_MIPS[vmType], DeepRLConstants.VM_PES[vmType]),
                    DeepRLConstants.SCHEDULING_INTERVAL));
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

            for (int j = 0; j < DeepRLConstants.HOST_PES[hostType]; j++) {
                peList.add(new Pe(j, new PeProvisionerSimple(DeepRLConstants.HOST_MIPS[hostType])));
            }

            hostList.add(new DRLHost(
                    i,
                    new RamProvisionerSimple(DeepRLConstants.HOST_RAM[hostType]),
                    new BwProvisionerSimple(DeepRLConstants.HOST_BW),
                    DeepRLConstants.HOST_STORAGE,
                    peList,
                    new VmSchedulerTimeSharedOverSubscription(peList),
                    new PowerModelSpecPowerDellPowerEdgeC6320(),
                    new CostModelAzure(Region.Australia_SouthEast, OS.Windows, Tier.Standard, Instance.A0)));
        }
        Log.print("Number of hosts created-> " + hostList.size());
        return hostList;
    }
}
