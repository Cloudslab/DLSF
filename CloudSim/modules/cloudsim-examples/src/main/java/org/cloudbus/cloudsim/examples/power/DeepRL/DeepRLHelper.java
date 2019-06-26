package org.cloudbus.cloudsim.examples.power.DeepRL;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.UtilizationModel;
import org.cloudbus.cloudsim.UtilizationModelNull;
import org.cloudbus.cloudsim.*;

import org.cloudbus.cloudsim.cost.models.CostModelAzure.*;
import org.cloudbus.cloudsim.examples.power.Constants;
import org.cloudbus.cloudsim.power.*;
import org.cloudbus.cloudsim.power.models.PowerModelSpecPowerDellPowerEdgeC6320;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

/**
 * @author Shreshth Tuli
 * @since June 16, 2019
 */

public class DeepRLHelper {

    public static int lastCloudletId = 0;

    public static int lastvmId = 0;

    public static int lastfileId = 0;

    public static Random rnd = new Random();

    public static int numCloudLets = 0;

    /**
     * Creates the cloudlet list bitbrain dataset
     *
     * @param brokerId the broker id
     * @param inputFolderName the input folder name
     * @return the list
     * @throws FileNotFoundException the file not found exception
     */
    public static List<DRLCloudlet> createCloudletListBitBrain(int brokerId, String inputFolderName)
            throws FileNotFoundException {
        List<DRLCloudlet> list = new ArrayList<DRLCloudlet>();

        long fileSize = 300;
        long outputSize = 300;
        int datasamples = DeepRLConstants.NUMBER_OF_DATA_SAMPLES;
        Log.printLine("@ " + DeepRLRunner.class.getSimpleName() + " inputFolder: " + inputFolderName);
        UtilizationModel utilizationModelNull = new UtilizationModelNull();

        File inputFolder = new File(inputFolderName);
        File[] files = inputFolder.listFiles();
        Log.printLine("@ " + DeepRLRunner.class.getSimpleName() + " inputFolder: " + inputFolder + " Number of files: " + files.length);

        for (int i = 0; i < DeepRLConstants.NUMBER_OF_VMS; i++) {
            DRLCloudlet cloudlet = null;
            try {
                System.out.println("@SSI- Filenumber- " + i + " filepath- " + files[i].getAbsolutePath() );
                cloudlet = new DRLCloudlet(
                        i,
                        (long)DeepRLConstants.SIMULATION_LIMIT,
                        DeepRLConstants.CLOUDLET_PES,
                        fileSize,
                        outputSize,
                        new UtilizationModelCPUBitBrainInMemory(files[i].getAbsolutePath(),DeepRLConstants.SCHEDULING_INTERVAL, datasamples),
                        new UtilizationModelRamBitBrainInMemory(files[i].getAbsolutePath(),DeepRLConstants.SCHEDULING_INTERVAL, datasamples),
                        new UtilizationModelNetworkRxBitBrainInMemory(files[i].getAbsolutePath(),DeepRLConstants.SCHEDULING_INTERVAL, datasamples),
                        new UtilizationModelDiskRxBitBrainInMemory(files[i].getAbsolutePath(),DeepRLConstants.SCHEDULING_INTERVAL, datasamples),
                        false,
                        0
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

    public static List<DRLCloudlet> createCloudletListBitBrainDynamic(int brokerId, String inputFolderName, int delay)
            throws FileNotFoundException {
        List<DRLCloudlet> list = new ArrayList<DRLCloudlet>();

        long fileSize = 300;
        long outputSize = 300;
        int datasamples = DeepRLConstants.NUMBER_OF_DATA_SAMPLES;
        Log.printLine("@ " + DeepRLRunner.class.getSimpleName() + " inputFolder: " + inputFolderName);
        UtilizationModel utilizationModelNull = new UtilizationModelNull();

        File inputFolder = new File(inputFolderName);
        File[] files = inputFolder.listFiles();
        Log.printLine("@ " + DeepRLRunner.class.getSimpleName() + " inputFolder: " + inputFolder + " Number of files: " + files.length);

        numCloudLets = (int) (rnd.nextGaussian() * DeepRLConstants.stdGaussian + DeepRLConstants.meanGaussian);
        numCloudLets = Math.max(numCloudLets, 0);

        for (int i = 0; i < numCloudLets; i++) {
            DRLCloudlet cloudlet = null;
            try {
                // lastfileId is the file index which circles around all files
                Log.printLine("@SSI- Filenumber- " + lastfileId + " filepath- " + files[lastfileId].getPath() );
                lastfileId = (lastfileId + 1) % files.length;

                // Cloudlet id and vm id are same and equal to lastCloudletId
                cloudlet = new DRLCloudlet(
                        lastCloudletId,
                        DeepRLConstants.CLOUDLET_LENGTH,
                        DeepRLConstants.CLOUDLET_PES,
                        fileSize,
                        outputSize,
                        new UtilizationModelCPUBitBrainInMemory(files[i].getAbsolutePath(),DeepRLConstants.SCHEDULING_INTERVAL, datasamples),
                        new UtilizationModelRamBitBrainInMemory(files[i].getAbsolutePath(),DeepRLConstants.SCHEDULING_INTERVAL, datasamples),
                        new UtilizationModelNetworkRxBitBrainInMemory(files[i].getAbsolutePath(),DeepRLConstants.SCHEDULING_INTERVAL, datasamples),
                        new UtilizationModelDiskRxBitBrainInMemory(files[i].getAbsolutePath(),DeepRLConstants.SCHEDULING_INTERVAL, datasamples),
                        false,
                        delay
                );
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(0);
            }
            cloudlet.setUserId(brokerId);
            cloudlet.setVmId(lastCloudletId);
            lastCloudletId = lastCloudletId + 1;
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
    public static List<Vm> createVmList(int brokerId, int vmsNumber, int delay) {
        List<Vm> vms = new ArrayList<Vm>();
        for (int i = 0; i < vmsNumber; i++) {
            int vmType = i / (int) Math.ceil((double) vmsNumber / DeepRLConstants.VM_TYPES);
            Log.printLine("Creating VM with VMID = " + lastvmId);
            vms.add(new DRLVm(
                    lastvmId,
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
                    DeepRLConstants.SCHEDULING_INTERVAL,
                    delay));
            lastvmId = lastvmId + 1;
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
                    new CostModelAzure(Region.Australia_SouthEast, OS.Windows, Tier.Standard, Instance.A0),
                    false));
        }
        Log.print("Number of hosts created-> " + hostList.size());
        return hostList;
    }

    /**
     * Creates the datacenter.
     *
     * @param name the name
     * @param datacenterClass the datacenter class
     * @param hostList the host list
     * @param vmAllocationPolicy the vm allocation policy
     *
     * @return the power datacenter
     *
     * @throws Exception the exception
     */
    public static Datacenter createDatacenter(
            String name,
            Class<? extends Datacenter> datacenterClass,
            List<PowerHost> hostList,
            VmAllocationPolicy vmAllocationPolicy, DRLDatacenterBroker broker) throws Exception {
        String arch = "x86"; // system architecture
        String os = "Linux"; // operating system
        String vmm = "Xen";
        double time_zone = 10.0; // time zone this resource located
        double cost = 3.0; // the cost of using processing in this resource
        double costPerMem = 0.05; // the cost of using memory in this resource
        double costPerStorage = 0.001; // the cost of using storage in this resource
        double costPerBw = 0.0; // the cost of using bw in this resource

        DatacenterCharacteristics characteristics = new DatacenterCharacteristics(
                arch,
                os,
                vmm,
                hostList,
                time_zone,
                cost,
                costPerMem,
                costPerStorage,
                costPerBw);

        Datacenter datacenter = null;
        try {
            datacenter = datacenterClass.getConstructor(
                    String.class,
                    DatacenterCharacteristics.class,
                    VmAllocationPolicy.class,
                    List.class,
                    Double.TYPE,
                    DRLDatacenterBroker
                            .class).newInstance(
                    name,
                    characteristics,
                    vmAllocationPolicy,
                    new LinkedList<Storage>(),
                    Constants.SCHEDULING_INTERVAL,
                    broker);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }

        return datacenter;
    }

}
