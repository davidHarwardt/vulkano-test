
with import <nixpkgs> {};
stdenv.mkDerivation rec {
    name = "vulkan";

    nativeBuildInputs = [
        pkg-config
    ];
    buildInputs = [
        vulkan-loader # vulkan
        xorg.libXcursor xorg.libXrandr xorg.libXi # x11
        libxkbcommon wayland # wayland

        cmake python3
        shaderc shaderc.lib shaderc.dev     # shaders
        shaderc.bin shaderc.static          # |
    ];

    
    LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
    # LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib:${pkgs.shaderc.lib}/lib:${pkgs.shaderc.dev}/lib";
    # VULKAN_LIB_DIR = "${pkgs.shaderc.dev}/lib";
    # shellHook = "echo ${pkgs.vulkan-loader}";
}

