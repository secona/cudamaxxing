{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
      in {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [ cmake ninja gnumake pkg-config ];

          buildInputs = with pkgs; [ 
            cudaPackages.cudatoolkit 
            cudaPackages.cuda_cudart
            linuxPackages.nvidia_x11
          ];

          packages = with pkgs; [ gcc12 ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH"
            export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
            export CUDACXX=${pkgs.cudaPackages.cudatoolkit}/bin/nvcc
          '';
        };
      });
}
