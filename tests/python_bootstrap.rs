use std::process::Command;
use std::sync::Once;

static PY_DEPS_INIT: Once = Once::new();

pub fn ensure_python_packages_installed() {
    PY_DEPS_INIT.call_once(|| {
        let packages = [
            ("numpy", "numpy"),
            ("scipy", "scipy"),
            ("scikit-learn", "sklearn"),
        ];

        let mut missing = Vec::new();
        for (package_name, module_name) in packages.iter() {
            let status = Command::new("python3")
                .args(["-c", &format!("import {}", module_name)])
                .status()
                .expect("failed to invoke python3 to probe optional modules");
            if !status.success() {
                missing.push(*package_name);
            }
        }

        if missing.is_empty() {
            return;
        }

        println!(
            "Installing missing Python packages required for reference PCA: {:?}",
            missing
        );

        let mut cmd = Command::new("python3");
        cmd.args(["-m", "pip", "install", "--user"]);
        cmd.args(&missing);

        let output = cmd
            .output()
            .expect("failed to invoke pip to install python dependencies");

        if !output.status.success() {
            panic!(
                "Failed to install required python packages. Status: {:?}\nSTDOUT:\n{}\nSTDERR:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    });
}
