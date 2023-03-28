# 很棒的WebAssembly Runtime ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)

这个项目包含了一系列执行WebAssembly（wasm）格式或将其编译为可执行机器码的虚拟机和工具 :octocat:

#### 图例 
:rocket: - 正在积极开发</br>
:sleeping: - 不再维护（距离上次提交已经超过一年）</br>

## 目录

- [aWsm](#awsm) <sup><sup>:rocket:</sup></sup></br>
- [CMM of Wasm](#cmm) <sup><sup>:sleeping:</sup></sup></br>
- [EOSVM](#eosvm) <sup><sup>:rocket:</sup></sup></br>
- [FDVM](#fdvm) <sup><sup>:sleeping:</sup></sup></br>
- [Fizzy](#fizzy) <sup><sup>:rocket:</sup></sup></br>
- [GraalWASM](#graalwasm) <sup><sup>:rocket:</sup></sup></br>
- [Happy New Moon With Report](#happy-new-moon-with-report) <sup><sup>:rocket:</sup></sup></br>
- [inNative](#innative) <sup><sup>:rocket:</sup></sup></br>
- [Life](#life) <sup><sup>:sleeping:</sup></sup></br>
- [Lucet](#lucet) <sup><sup>:rocket:</sup></sup></br>
- [wasm3](#wasm3) <sup><sup>:rocket:</sup></sup></br>
- [Motor](#motor) <sup><sup>:sleeping:</sup></sup></br>
- [py-wasm](#py-wasm) <sup><sup>:sleeping:</sup></sup></br>
- [Serverless Wasm](#serverless-wasm) <sup><sup>:sleeping:</sup></sup></br>
- [Swam](#swam) <sup><sup>:rocket:</sup></sup></br>
- [VMIR](#vmir) <sup><sup>:sleeping:</sup></sup></br>
- [wac](#wac) <sup><sup>:sleeping:</sup></sup></br>
- [Wagon](#wagon) <sup><sup>:sleeping:</sup></sup></br>
- [WAKit](#wakit) <sup><sup>:rocket:</sup></sup></br>
- [Warpy](#warpy) <sup><sup>:sleeping:</sup></sup></br>
- [WasmEdge](#wasmedge) <sup><sup>:rocket:</sup></sup></br>
- [Wasmer](#wasmer) <sup><sup>:rocket:</sup></sup></br>
- [Wasmi](#wasmi) <sup><sup>:rocket:</sup></sup></br>
- [Wasmo](#wasmo) <sup><sup>:rocket:</sup></sup></br>
- [WasmRT](#wasmrt) <sup><sup>:sleeping:</sup></sup></br>
- [Wasmtime](#wasmtime) <sup><sup>:rocket:</sup></sup></br>
- [WasmVM](#wasmvm1) <sup><sup>:rocket:</sup></sup></br>- [WasmVM（WebAssembly虚拟机）](#wasmvm2) <sup><sup>:sleeping:</sup></sup></br>
- [WAVM（WebAssembly虚拟机）](#wavm) <sup><sup>:rocket:</sup></sup></br>
- [WebAssembly（WebAssembly）](#webassembly) <sup><sup>:sleeping:</sup></sup></br>
- [WebAssembly Micro Runtime（WebAssembly微运行时）](#wamr) <sup><sup>:rocket:</sup></sup></br>
- [TWVM](#twvm) <sup><sup>:rocket:</sup></sup></br>
- [wazero](#wazero) <sup><sup>:rocket:</sup></sup></br>

----------------


## <a name="awsm"></a>[aWsm](https://github.com/gwsystems/aWsm) <sup>[top⇈](#contents)</sup>

> aWsm是将WebAssembly（Wasm）代码编译为llvm字节码，然后编译为在各种平台上运行的沙箱二进制文件的编译器和运行时。

* **编写语言**

    <table>
    <tr>
        <td>Rust</td>
        <td>C</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>LLVM</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>AOT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    <table>
    <tr>
        <td>C</td>
    </tr>
    </table>

* **非MVP功能支持**

    - `无`

* **支持的主机API**

    - `无`

* **非Web标准**

    - `无`

* **应用**

    - `无`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
    </tr>
    </table>


## <a name="cmm"></a>[CMM of Wasm](https://github.com/SimonJF/cmm_of_wasm) <sup>[top⇈](#contents)</sup>
> 一种从WebAssembly到本地代码的编译器，通过OCaml后端。

* **编写语言**

    <table>
    <tr>
        <td>OCaml</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>OCaml编译器</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>AOT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `无`

* **非MVP功能支持**

    - `无`

* **支持的主机API**

    - `无`

* **非Web标准**

    - `无`

* **应用**

    - `无`

* **支持的平台**

    <table><tr>
    <td>Linux</td>
    <td>macOS</td>
</tr>
</table>

## <a name="eosvm"></a>[EOSVM](https://github.com/EOSIO/eos-vm) <sup>[返回顶部⇈](#contents)</sup>
> EOS VM是从最基础的方面设计的，以满足区块链应用的高要求，而这种要求要比那些为 Web 浏览器或标准开发设计的 WebAssembly 引擎要高出很多。

* **编写语言**

    <table>
    <tr>
        <td>C++</td>
    </tr>
    </table>

* **编译框架**

    <table>
    <tr>
        <td>Custom</td>
    </tr>
    </table>


* **编译 / 执行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非 MVP 特性**

    - `N/A`

* **支持的主机 API**

    - `N/A`

* **支持的非 Web 标准**

    - `N/A`

* **被使用的项目**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="fdvm"></a>[FDVM](https://github.com/funcdef/fdvm) <sup>[返回顶部⇈](#contents)</sup>
> 用于开发服务器端 WebAssembly 应用程序的 WASM 运行时。

* **编写语言**

    <table>
    <tr>
        <td>JavaScript</td>
    </tr>
    </table>

* **编译框架**

    <table>
    <tr>
        <td>NodeJS(V8)</td>
    </tr>
    </table>


* **编译 / 执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非 MVP 特性**

    - `N/A`

* **支持的主机 API**

    - `N/A`

* **支持的非 Web 标准**

    - `N/A`

* **被使用的项目**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="fizzy"></a>[Fizzy](https://github.com/wasmx/fizzy) <sup>[返回顶部⇈](#contents)</sup>
> Fizzy 旨在成为一个快速、确定性和一丝不苟的 C++ WebAssembly 解释器。* **编程语言**

    <table>
    <tr>
        <td>C++</td>
    </tr>
    </table>

* **编译器框架**

    - `N/A`

* **编译/执行模式**

    <table>
    <tr>
        <td>解释器</td>
    </tr>
    </table>

* **与其他编程语言的互操作性**

    - `N/A`

* **支持的非MVP功能**

    - `N/A`

* **支持的主机API**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **非Web标准**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
    </tr>
    </table>

## <a name="graalwasm"></a>[GraalWasm](https://github.com/oracle/graal/tree/master/wasm) <sup>[回到顶部⇈](#contents)</sup>
GraalWasm是实现于GraalVM中的WebAssembly引擎。它可以解释和编译二进制格式的WebAssembly程序，或嵌入到其他程序中。

* **编程语言**

    <table>
    <tr>
        <td>Java</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>GraalVM</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他编程语言的互操作性**

    <table>
    <tr>
        <td>Java</td>
        <td>JVM</td>
        <td>支持Graal的编程语言</td>
    </tr>
    </table>

* **支持的非MVP功能**

    <table>
    <tr>
    </tr>
    </table>

* **支持的主机API**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **非Web标准**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **使用者**

    - [GraalVM JavaScript](https://github.com/graalvm/graaljs) - 高性能的JavaScript编程语言实现。

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>## <a name="happy-new-moon-with-report"></a>[快乐新月报告](https://github.com/fishjd/HappyNewMoonWithReport) <sup>[返回⇈](#contents)</sup>

快乐新月报是完全使用Java编写的WebAssembly的开源实现。它用于在Java中运行或测试Web Assembly模块(*.wasm)。

* **开发语言**

    <table>
    <tr>
        <td>Java</td>
    </tr>
    </table>

* **编译器框架**

    - `N/A`

* **编译/执行模式**

    <table>
    <tr>
        <td>解释模式</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    <table>
    <tr>
        <td>Java</td>
        <td>JVM编程语言</td>
    </tr>
    </table>

* **支持的非MVP功能**

    <table>
    <tr>
    	<td>符号扩展</td>
    </tr>
    </table>

* **支持的托管API**

    - `N/A`

* **使用者**

    - `N/A`


* **支持的平台**

    <table>
    <tr>
        <td>JVM</td>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="innative"></a>[inNative](https://github.com/innative-sdk/innative) <sup>[返回⇈](#contents)</sup>
>An AOT (ahead-of-time) compiler for WebAssembly that creates C compatible binaries, either as sandboxed plugins you can dynamically load, or as stand-alone executables that interface directly with the operating system.


* **开发语言**

    <table>
    <tr>
        <td>C++</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>LLVM</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>AOT</td>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非MVP功能**

    <table>
    <tr>
        <td>线程</td>
        <td>多项结果和块参数</td>
        <td>名称部分</td>
        <td>批量内存操作</td>
        <td>符号扩展指令</td><td>非陷阱转换指令</td>
    </tr>
    </table>

* **支持的主机API**
    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>

* **非Web标准**
    - `N/A`

* **使用者**
    - `N/A`

* **支持的平台**
    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="life"></a>[Life](https://github.com/perlin-network/life) <sup>[返回顶部⇈](#contents)</sup>
> Life是一个安全而快速的WebAssembly虚拟机，用于去中心化应用程序，由Perlin Network用Go语言编写。

* **编写语言**
    <table>
    <tr>
        <td>Go</td>
    </tr>
    </table>
    
* **编译器框架**
    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>
    
* **编译/执行模式**
    <table>
    <tr>
        <td>解释型</td>
    </tr>
    </table>
    
* **与其他语言的互操作性**
    - `N/A`
    
* **支持非MVP功能**
    - `N/A`
    
* **支持的主机API**
    - `N/A`
    
* **非Web标准**
    - `N/A`
    
* **使用者**
    - `N/A`
    
* **支持的平台**
    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

    
## <a name="lucet"></a>[Lucet](https://github.com/fastly/lucet) <sup>[返回顶部⇈](#contents)</sup>
> Lucet是一种本地WebAssembly编译器和运行时。它旨在安全地在应用程序内执行不受信任的WebAssembly程序。

> 注意：Lucet现处于维护模式。工作已移至[wasmtime](https://github.com/bytecodealliance/wasmtime/)。

* **编写语言**
    <table>
    <tr>
        <td>Rust</td>
    </tr>
    </table>
    
* **编译器框架**
    <table>
    <tr>
        <td>Cranelift</td>
    </tr>
    </table>
    
* **编译/执行模式**
    <table>
    <tr>
        <td>AOT</td>
    </tr>
    </table>
    
* **与其他语言的互操作性**
    - `N/A`
    
* **支持非MVP功能**
    - `N/A`
    
* **支持的主机API**
    - `N/A`
    
* **非Web标准**
    - `N/A`
    
* **使用者**
    - `N/A`
    
* **支持的平台**".<table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **非 Web 标准**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
    </tr>
    </table>


## <a name="wasm3"></a>[Wasm3](https://github.com/wasm3/wasm3) <sup>[返回顶部⇈](#contents)</sup>
> 🚀 高性能的 WebAssembly 解释器

* **编写语言**

    <table>
    <tr>
        <td>C</td>
    </tr>
    </table>

* **编译框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译 / 执行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    <table>
    <tr>
        <td>Python</td>
        <td>C/C++</td>
        <td>Rust</td>
        <td>Go</td>
        <td>Zig</td>
        <td>Swift</td>
        <td>C#/.Net</td>
    </tr>
    </table>

* **支持的非 MVP 特性**

    <table>
    <tr>
        <td>多值</td>
        <td>批量内存操作</td>
        <td>符号扩展操作符</td>
        <td>非陷阱转换</td>
        <td>Name Section</td>
    </tr>
    </table>

* **支持的主机 API**

    <table>
    <tr>
        <td>WASI</td>
        <td>自定义</td>
    </tr>
    </table>

* **非 Web 标准**

    <table>
    <tr>
        <td>WASI</td>
        <td>燃气计量</td>
    </tr>
    </table>

* **使用者**

    - [wasmcloud](https://wasmcloud.dev/) - 一个编写便携商业逻辑的平台
    - [Shareup](https://shareup.app/) - 为每个人提供快速、私密的分享
    - [WowCube](https://wowcube.com/) - 一个创新的主机和游戏平台
    - [txiki.js](https://github.com/saghul/txiki.js) - 一个小而强大的 JavaScript 运行时

* **支持的平台**

    <table>
    <tr>
        <td>Windows</td>
        <td>Linux<br/>(任何架构)</td>
        <td>macOS<br/>(任何架构)</td>
        <td>FreeBSD<br/>(任何架构)</td>
        <td>Android</td>```
<table>
    <tr>
        <td>OpenWRT</td>
        <td>SBC/MCU</td>
        <td>Arduino</td>
    </tr>
</table>

## <a name="motor"></a>[Motor](https://github.com/penberg/motor) <sup>[返回顶部⇈](#contents)</sup>
> Motor是一个WebAssembly运行时，旨在安全高效地执行WebAssembly程序。

* **编程语言**

    <table>
    <tr>
        <td>Rust</td>
    </tr>
    </table>

* **编译框架**

    <table>
    <tr>
        <td>Dynasm</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他编程语言互操作性**

    - `N/A`

* **支持的非MVP特性**

    - `N/A`

* **支持的主机API**

    - `N/A`

* **非Web标准**

    - `N/A`

* **被使用**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
    </tr>
    </table>


## <a name="py-wasm"></a>[py-wasm](https://github.com/ethereum/py-wasm) <sup>[返回顶部⇈](#contents)</sup>
> WebAssembly解释器的Python实现

* **编程语言**

    <table>
    <tr>
        <td>Python</td>
    </tr>
    </table>

* **编译框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他编程语言互操作性**

    - `N/A`

* **支持的非MVP特性**

    - `N/A`

* **支持的主机API**

    - `N/A`

* **非Web标准**

    - `N/A`

* **被使用**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="serverless-wasm"></a>[Serverless wasm](https://github.com/Geal/serverless-wasm) <sup>[返回顶部⇈](#contents)</sup>
> 展示WebAssembly在浏览器以外的潜力的一个小演示

* **编程语言**

    <table>
    <tr>
        <td>Rust</td>
    </tr>
    </table>

* **编译框架**

    - `N/A`

* **编译/执行模式**

    - `N/A`

* **与其他编程语言互操作性**

    - `N/A`

* **支持的非MVP特性**

    - `N/A`

* **支持的主机API**

    - `N/A`

* **非Web标准**

    - `N/A`

* **被使用**

    - `N/A`

* **支持的平台**

    - `N/A`
<table>
    <tr>
        <td>Wasmi</td>
        <td>Cranelift</td>
    </tr>
</table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释型</td>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非MVP功能**

    - `N/A`

* **支持的宿主API**

    - `N/A`

* **非Web标准**

    - `N/A`

* **使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="swam"></a>[Swam](https://github.com/satabin/swam) <sup>[top⇈](#contents)</sup>
> Scala的WebAssembly引擎

* **使用的语言**

    <table>
    <tr>
        <td>Scala</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>定制框架</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释型</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非MVP功能**

    - `N/A`

* **支持的宿主API**

    - `N/A`

* **非Web标准**

    - `N/A`

* **使用者**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="vmir"></a>[VMIR](https://github.com/andoma/vmir) <sup>[top⇈](#contents)</sup>
> VMIR是一个独立的库，它是用C编写的，可以解析和执行WebAssembly和LLVM位代码文件

* **使用的语言**

    <table>
    <tr>
        <td>C</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>LLVM</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释型</td>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非MVP功能**

    - `N/A`

* **支持的宿主API**

    - `N/A`

* **非Web标准**.- `N/A`

* **使用人群**

    - `N/A`
    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="wac"></a>[wac](https://github.com/kanaka/wac) <sup>[返回顶部⇈](#contents)</sup>
> 用C语言编写的最小WebAssembly解释器。

* **编写语言**

    <table>
    <tr>
        <td>C</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>Custom</td>
    </tr>
    </table>


* **编译/ 执行模式**

    <table>
    <tr>
        <td>解释型</td>
    </tr>
    </table>

* **与其他语言互操作性**

    - `N/A`

* **支持的非MVP特性**

    - `N/A`

* **支持的应用程序接口**

    - `N/A`

* **非Web标准**

    - `N/A`

* **使用人群**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="wagon"></a>[wagon](https://github.com/go-interpreter/wagon) <sup>[返回顶部⇈](#contents)</sup>
> Wagon 是一个基于 WebAssembly 的 Go 解释器，用于 Go。

* **编写语言**

    <table>
    <tr>
        <td>Go</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>Custom</td>
    </tr>
    </table>


* **编译/ 执行模式**

    <table>
    <tr>
        <td>解释型</td>
    </tr>
    </table>

* **与其他语言互操作性**

    - `N/A`

* **支持的非MVP特性**

    - `N/A`

* **支持的应用程序接口**

    - `N/A`

* **非Web标准**

    - `N/A`

* **使用人群**

    - `N/A`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="wakit"></a>[WAKit](https://github.com/akkyie/WAKit) <sup>[返回顶部⇈](#contents)</sup>
> 用 Swift 编写的 WebAssembly 运行时。

* **编写语言**

    <table>
    <tr>
        <td>Swift</td>
    </tr>
    </table>* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释器</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `无`

* **不支持MVP的特性**

    - `无`

* **支持的宿主API**

    - `无`

* **非Web标准**

    - `无`

* **使用者**

    - `无`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
    </tr>
    </table>

## <a name="warpy"></a>[Warpy](https://github.com/kanaka/warpy) <sup>[返回顶部⇈](#contents)</sup>
> RPython中的WebAssembly解释器。

* **编写的语言**

    <table>
    <tr>
        <td>RPython</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>RPython</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `无`

* **不支持MVP的特性**

    - `无`

* **支持的宿主API**

    - `无`

* **非Web标准**

    - `无`

* **使用者**

    - `无`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="wasmedge"></a>[WasmEdge](https://github.com/WasmEdge/WasmEdge) <sup>[返回顶部⇈](#contents)</sup>
> 用于云原生、边缘和分布式应用程序的轻量级、高性能、可扩展的WebAssembly运行时。属于CNCF项目。

* **编写的语言**

    <table>
    <tr>
        <td>C++</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>LLVM</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释器</td>
        <td>AOT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    <table>
    <tr>
        <td>Solidity</td>
        <td>Rust</td>
        <td>C/C++</td>
    </tr>
    </table><td>Go/TinyGo</td>
        <td>JavaScript</td>
        <td>Python</td>
        <td>Grain</td>
        <td>Swift</td>
        <td>Zig</td>
        <td>Ruby</td>
    </tr>
    </table>

* **支持非MVP功能**

    <table>
    <tr>
        <td>批量内存操作</td>
        <td>SIMD</td>
        <td>多内存</td>
        <td>多值</td>
        <td>引用类型</td>
        <td>有符号扩展指令</td>
        <td>非陷阱浮点到整数转换</td>
        <td>可变全局变量</td>
    </tr>
    </table>

* **支持主机API**

    <table>
    <tr>
        <td>WASI</td>
        <td>网络套接字</td>
        <td>TensorFlow</td>
        <td>命令接口</td>
        <td>图像处理</td>
    </tr>
    </table>

* **非Web标准**

    <table>
    <tr>
        <td>WASI</td>
        <td>燃气计量</td>
        <td>Ethereum环境接口</td>
        <td>Oasis</td>
        <td>Substrate</td>
    </tr>
    </table>

* **被以下项目使用**

    - [Suborbital](https://blog.suborbital.dev/suborbital-wasmedge)
    - [crun](https://github.com/containers/crun/pull/774)
    - [SuperEdge](https://github.com/superedge/superedge/pull/335)
    - [OpenYurt](https://github.com/openyurtio/openyurt.io/pull/85)
    - [Dapr](https://www.infoq.com/articles/webassembly-dapr-wasmedge/)
    - [Yomo](https://github.com/yomorun/yomo-wasmedge-tensorflow)
    - [Oasis ETH ParaTime](https://medium.com/oasis-protocol-project/the-oasis-eth-paratime-is-live-on-mainnet-33d8713ec870)

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
        <td>Android</td>
        <td>OpenHarmony</td>
        <td>seL4 RTOS</td>
    </tr>
    </table>

## <a name="wasmer"></a>[Wasmer](https://github.com/wasmerio/wasmer) <sup>[返回目录](#contents)</sup>
> Wasmer 是一个独立的 WebAssembly 运行时，用于在浏览器之外运行 WebAssembly，支持 WASI 和 Emscripten。

* **使用的语言**

    <table>
    <tr><td>锈</td>
        <td>C++</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>Singlepass（单遍）</td>
        <td>Cranelift（默认）</td>
        <td>LLVM</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>JIT</td>
        <td>AOT</td>
    </tr>
    </table>

* **与其他语言互操作性**

    <table>
    <tr>
        <td>锈</td>
        <td>PHP</td>
        <td>C</td>
        <td>C++</td>
        <td>Python</td>
        <td>Go</td>
        <td>PHP</td>
        <td>Java</td>
        <td>Ruby</td>
        <td>Postgres</td>
        <td>C#/.Net</td>
        <td>Elixir</td>
        <td>R</td>
        <td>D</td>
        <td>Swift</td>
    </tr>
    </table>

* **支持的非MVP特性**

    <table>
    <tr>
        <td>Threads（线程）</td>
        <td>SIMD</td>
        <td>Multi Value returns（多值返回）</td>
        <td>Name Section（名称段）</td>
        <td>Bulk Memory Operations（批量内存操作）</td>
        <td>Sign Extension Instructions（符号扩展指令）</td>
    </tr>
    </table>

* **支持的Host APIs**

    <table>
    <tr>
        <td>Emscripten</td>
        <td>WASI</td>
    </tr>
    </table>

* **非Web标准**

    <table>
    <tr>
        <td>WASI</td>
        <td>wasm-c-api</td>
    </tr>
    </table>

* **被以下项目使用**

    - [Spacemesh虚拟机](https://github.com/spacemeshos/svm) - 用于Spacemesh智能合约的虚拟机
    - [CosmWasm](https://github.com/CosmWasm/cosmwasm) - 能够运行wasm智能合约的Cosmos兼容库
    - [NEAR Protocol](https://github.com/near/nearcore) - NEAR协议的参考客户端
    - [Dprint](https://github.com/dprint/dprint) - 可插拔且可配置的代码格式化平台

* **支持的平台**

    <table>
    <tr>
        <td>Linux（x64，aarch64，x86）</td>
        <td>macOS（x64，arm64）</td>
        <td>Windows（x64，x86）</td>
        <td>FreeBSD（x64）</td>
        <td>Android</td>
    </tr>
    </table>## <a name="wasmi"></a>[Wasmi](https://github.com/paritytech/wasmi) <sup>[返回⇈](#contents)</sup>
> 一个Wasm解释器。

* **编程语言**

    <table>
    <tr>
        <td>Rust</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译/执行模式**
    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他语言的互操作性**
    - `无`

* **支持的非MVP功能**
    - `无`

* **支持的主机API**
    - `无`

* **非Web标准**
    - `无`

* **应用**
    - `无`

* **平台支持**
    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="wasmo"></a>[Wasmo](https://github.com/appcypher/wasmo) <sup>[返回⇈](#contents)</sup>
> 基于LLVM的WebAssembly运行时和编译器。

* **编程语言**

    <table>
    <tr>
        <td>Rust</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>LLVM</td>
    </tr>
    </table>


* **编译/执行模式**
    <table>
    <tr>
        <td>JIT即时编译</td>
    </tr>
    </table>

* **与其他语言的互操作性**
    - `无`

* **支持的非MVP功能**
    - `无`

* **支持的主机API**
    - `无`

* **非Web标准**
    - `无`

* **应用**
    - `无`

* **平台支持**
    <table>
    <tr>
        <td>macOS</td>
    </tr>
    </table>

## <a name="wasmrt"></a>[wasmrt](https://github.com/rhitchcock/wasmrt) <sup>[返回⇈](#contents)</sup>
> wasmrt是专为本地执行WebAssembly模块构建的运行时（在最初进行虚拟化，最终进行JIT编译）。

* **编程语言**

    <table>
    <tr>
        <td>C ++</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译/执行模式**
    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>* **与其他语言的互操作性**

    - `无`

* **支持的非MVP功能**

    - `无`

* **支持的主机API**

    - `无`

* **非Web标准**

    - `无`

* **使用者**

    - `无`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="wasmtime"></a>[Wasmtime](https://github.com/CraneStation/wasmtime) <sup>[返回顶部⇈](#contents)</sup>
> Wasmtime是一个独立的WebAssembly运行时，只使用Cranelift JIT

* **编写的语言**

    <table>
    <tr>
        <td>Rust</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>Cranelift</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    <table>
    <tr>
        <td>Python</td>
    </tr>
    </table>

* **支持的非MVP功能**

    <table>
    <tr>
        <td>接口类型</td>
    </tr>
    </table>

* **支持的主机API**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **非Web标准**

    <table>
    <tr>
        <td>WASI</td>
        <td>wasm-c-api</td>
    </tr>
    </table>

* **使用者**

    - `无`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="wasmvm1"></a>[WasmVM](https://github.com/WasmVM/WasmVM) <sup>[返回顶部⇈](#contents)</sup>
> 一款非官方的独立WebAssembly进程虚拟机

* **编写的语言**

    <table>
    <tr>
        <td>C++</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `无`

* **支持的非MVP功能**

    - `无`* **支持的主机API**

    - `不适用`

* **非web标准**

    - `不适用`

* **使用者**

    - `不适用`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="wasmvm2"></a>[wasmvm](https://github.com/kogai/wasvm) <sup>[返回顶部⇈](#contents)</sup>
> WebAssembly虚拟机，旨在适配嵌入式系统。

* **编写语言**

    <table>
    <tr>
        <td>Rust</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>Life</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `不适用`

* **支持的非MVP特性**

    - `不适用`

* **支持的主机API**

    - `不适用`

* **非web标准**

    - `不适用`

* **使用者**

    - `不适用`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="wavm"></a>[WAVM](https://github.com/WAVM/WAVM) <sup>[返回顶部⇈](#contents)</sup>
> WebAssembly虚拟机，旨在适配嵌入式系统。

* **编写语言**

    <table>
    <tr>
        <td>C++</td>
        <td>Python</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>LLVM</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>JIT(即时编译)</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `不适用`

* **支持的非MVP特性**

    <table>
    <tr>
        <td>线程模式</td>
        <td>SIMD</td>
        <td>多重结果和块参数</td>
        <td>异常处理</td>
        <td>名字段(Name Section)</td>
        <td>引用类型</td>
        <td>批量内存操作</td>
        <td>符号扩展指令</td>
    </tr>
    </table>

* **支持的主机API**

    <table>
    <tr>
        <td>WASI</td><td>Emscripten</td>
        <td>WAVIX</td>
    </tr>
    </table>

* **非Web标准**

    - [x] WASI

* **使用者**

    - `尚不适用`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>

## <a name="webassembly"></a>[webassembly](https://github.com/dcodeIO/webassembly) <sup>[top⇈](#contents)</sup>
> 一种实验性的最小工具集和运行时，用于在node之上生成和运行WebAssembly模块

* **编写的语言**

    <table>
    <tr>
        <td>JavaScript</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>NodeJS(V8)</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `尚不适用`

* **支持的非MVP特性**

    - `尚不适用`

* **支持的主机API**

    - `尚不适用`

* **非Web标准**

    - `尚不适用`

* **使用者**

    - `尚不适用`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>


## <a name="wamr"></a>[WAMR](https://github.com/bytecodealliance/wasm-micro-runtime) <sup>[top⇈](#contents)</sup>
> WebAssembly Micro Runtime (WAMR)是一个独立的WebAssembly (WASM)运行时，具有很小的占用空间

* **编写的语言**

    <table>
    <tr>
        <td>C</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释执行</td>
        <td>AOT-模块</td>
        <td>JIT</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `尚不适用`

* **支持的非MVP特性**

    - [非限制性的浮点整数转换](https://github.com/WebAssembly/nontrapping-float-to-int-conversions)
    - [符号扩展运算符](https://github.com/WebAssembly/sign-extension-ops)`- [内存块操作](https://github.com/WebAssembly/bulk-memory-operations)
    - [共享内存](https://github.com/WebAssembly/threads/blob/main/proposals/threads/Overview.md#shared-linear-memory)
    - [多返回值](https://github.com/WebAssembly/multi-value)
    - [尾调用](https://github.com/WebAssembly/tail-call)


* **支持的主机API**

    - [wasm-c-api](https://github.com/WebAssembly/wasm-c-api)

* **非Web标准**

    - [x] WASI

* **使用者**

    - `不适用`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
        <td>AliOS-Things</td>
        <td>Android</td>
        <td>Linux-SGX</td>
        <td>Nuttx</td>
        <td>VxWorks</td>
        <td>Zephyr</td>
    </tr>
    </table>



## <a name="twvm"></a>[TWVM](https://github.com/Becavalier/TWVM) <sup>[返回顶部⇈](#contents)</sup>
> 一个小型高效的WebAssembly虚拟机。

* **编写语言**

    <table>
    <tr>
        <td>C++</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr>
        <td>解释执行</td>
    </tr>
    </table>

* **与其他语言的互操作性**

    - `不适用`

* **支持的非MVP功能**

    - `不适用`

* **支持的主机API**

    - `不适用`

* **非Web标准**

    - `不适用`

* **使用者**

    - `不适用`

* **支持的平台**

    <table>
    <tr>
        <td>Linux</td>
        <td>macOS</td>
        <td>Windows</td>
    </tr>
    </table>



## <a name="wazero"></a>[wazero](https://wazero.io) <sup>[返回顶部⇈](#contents)</sup>
> wazero是一个符合WebAssembly 1.0和2.0标准的运行时，使用Go语言编写，且不依赖任何平台。

* **编写语言**

    <table>
    <tr>
        <td>Go</td>
    </tr>
    </table>

* **编译器框架**

    <table>
    <tr>
        <td>自定义</td>
    </tr>
    </table>


* **编译/执行模式**

    <table>
    <tr><td>解释执行</td>
<td>JIT</td>
</tr>
</table>

* **与其他语言的互操作性**

    - `N/A`

* **支持的非 MVP 功能**

    <table>
    <tr>
        <td>批量内存操作</td>
        <td>可变全局变量的导入/导出</td>
        <td>符号扩展运算符</td>
        <td>多值返回</td>
        <td>名称部分</td>
        <td>非陷阱的浮点数转整数转换</td>
        <td>引用类型</td>
        <td>SIMD</td>
    </tr>
    </table>

* **支持的宿主 API**

    <table>
    <tr>
        <td>AssemblyScript</td>
        <td>WASI</td>
    </tr>
    </table>

* **非 Web 标准**

    <table>
    <tr>
        <td>WASI</td>
    </tr>
    </table>

* **被以下项目使用**

    - [dapr](https://github.com/dapr/dapr) - 简化微服务连接的 API
    - [trivy](https://github.com/aquasecurity/trivy) - 面向容器和其他构件的漏洞/错误配置/机密扫描器
    - [wapc-go](https://github.com/wapc/wapc-go) - wapc 兼容模块的 WebAssembly 主机运行时

* **支持的平台**

    <table>
    <tr>
        <td>FreeBSD（amd64）</td>
        <td>Linux（amd64、arm64、riscv64）</td>
        <td>macOs（amd64）</td>
        <td>Windows（amd64）</td>
    </tr>
    </table>
-------------------

## 许可证

[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

在法律范围内，[Steve Akinyemi](https://github.com/appcypher)已放弃对此作品的所有版权和相关或邻近的权利。
