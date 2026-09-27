#ifndef TCP_SERVER_H
#define TCP_SERVER_H

#include <cstdint>
#include <string>
#include <vector>

class TcpServer
{
public:
    TcpServer(const std::string& ip, uint16_t port);
    ~TcpServer();

    bool start();
    bool acceptClient();

    bool receiveMessage(std::vector<uint8_t>& data);
    bool sendMessage(const std::vector<uint8_t>& data);

    void closeClient();
    void stop();

private:
    bool sendAll(const uint8_t* data, size_t size);
    bool receiveAll(uint8_t* data, size_t size);

private:
    std::string ip_;
    uint16_t port_;

    int server_fd_;
    int client_fd_;
};

#endif
