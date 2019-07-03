package org.cloudbus.cloudsim.power;

import org.cloudbus.cloudsim.Vm;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class DRLVmSelectionPolicy extends PowerVmSelectionPolicy{

    public int migrationPenalty = 0;

    /**
     * Gets the list of migratable VMs from a given host.
     *
     * @return the list of migratable VMs
     */
    protected List<PowerVm> getAllMigratableVms(List<PowerHost> hosts) {
        List<PowerVm> migratableVms = new ArrayList<PowerVm>();
        for(PowerHost host : hosts){
            for (PowerVm vm : host.<PowerVm> getVmList()) {
                if (!vm.isInMigration()) {
                    migratableVms.add(vm);
                }
            }
        }
        return migratableVms;
    }

    /**
     * Gets a VM to migrate from a given host.
     *
     * @param host the host
     * @return the vm to migrate
     */
    @Override
    public Vm getVmToMigrate(PowerHost host){
        List<Vm> output = new LinkedList<Vm>();
        List<Vm> vmsToMigrate = new LinkedList<Vm>();
        // Parse output from DL model
        for (Vm vm : output) {
            if(!vm.isInMigration()){
                vmsToMigrate.add(vm);
            }
        }
        return ((LinkedList<Vm>) vmsToMigrate).getFirst();
    }

    /**
     * Gets the VMs to migrate from hosts.
     *
     * @return the VMs to migrate from hosts
     */
    protected List<? extends Vm>
    getAllVmsToMigrate(List<PowerHost> hosts, List<? extends Vm> vmList) {
        List<DRLVm> vmsToMigrate = new LinkedList<DRLVm>();
        List<PowerVm> migratableVms = getAllMigratableVms(hosts);
        // @todo : Parse output from DL model
        String result = ""; ArrayList<String> vms = new ArrayList();
        try{
            DRLDatacenter.toPython.println(("getVmsToMigrate\nEND")); DRLDatacenter.toPython.flush();
            result = DRLDatacenter.fromPython.readLine();
            vms = new ArrayList<>(Arrays.asList(result.split(" ")));
        }
        catch(Exception e){
            System.out.println(e.getMessage());
        }
        migrationPenalty = 0;
        DRLVm vm = (DRLVm)vmList.get(0);
        for (int i = 0; i < vms.size(); i++) {
            try{
                vm = (DRLVm)vmList.get(Integer.parseInt(vms.get(i)));
            }
            catch(Exception e){
                System.out.println("Error in integer parse: " + vms.size() + ", " + vms.get(i) + ", -" + result + "-");
                System.exit(0);
            }
            if(migratableVms.contains(vm)){
                vmsToMigrate.add(vm);
            }
            else{
                migrationPenalty += 1;
            }
        }
        return vmsToMigrate;
    }
}
